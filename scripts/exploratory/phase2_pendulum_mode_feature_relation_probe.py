#!/usr/bin/env python
"""Relate prior environment features to the purified Pendulum control mode.

Read-only exploratory probe.  It consumes fixed-list h files and labels from
the long-horizon mode extraction; it does not sample, train, or mutate the
prior generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MANIFESTS = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/full/selector_probe_manifest.json",
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/q90/selector_probe_manifest.json",
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_pendulum_mode_feature_relation_probe_0509"

TARGETS = (
    "closed_loop_core",
    "primary_long_horizon_target",
    "selector_hard_positive",
    "decay_settle_core",
)

PROTECTED_DIM_FIELDS = {"action_dim", "obs_dim", "state_dim", "noise_dim", "zero_pad_dim"}
REWARD_PATH_FIELDS = {
    "reward_state_input_gain_fraction_conditioned_min",
    "reward_action_input_gain_fraction_conditioned_min",
    "reward_state_to_action_gain_ratio_conditioned_max",
    "reward_state_input_gain_fraction_rejection_min",
    "reward_topology_conditioned_sampling_enabled",
    "reward_topology_conditioned_sampling_max_attempts",
    "exploratory_reward_group_state_scale",
    "exploratory_reward_group_action_scale",
    "exploratory_reward_group_noise_scale",
    "exploratory_reward_group_balance_enabled",
    "exploratory_reward_group_balance_profile",
}
CONTROL_COST_FIELDS = {
    "ctrl_reward_weight",
    "ctrl_reward_enable_prob",
    "_ctrl_reward_enable_u",
    "reward_scale",
}
TRANSITION_CONTROL_FIELDS = {
    "alpha",
    "init_action_std",
    "init_state_std",
    "init_std",
    "state_noise_std",
    "state_highway_enabled",
    "state_highway_lambda",
    "state_input_scale",
    "state_input_scale_enabled",
    "state_output_scale",
    "lengthscale",
    "outputscale",
    "reference_state_inertia_enabled",
    "strict_joint_transition_enabled",
}
TERMINAL_SURVIVAL_FIELDS = {
    "terminal_reset_enabled",
    "terminal_reset_count_target",
    "terminal_bonus_scale_min",
    "terminal_bonus_scale_max",
    "terminal_bonus_tanh_c",
    "survival_reward_enable_prob",
    "survival_reward_weight",
}
TRAINING_OR_PPO_FIELDS_PREFIXES = (
    "reinforce_",
    "pg_",
    "policy_gradient_",
    "normalized_q_",
    "next_state_flow_",
    "batch_",
    "transition_",
    "reference_scm_",
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    value = obj
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


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
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _feature_group(name: str) -> str:
    if name in PROTECTED_DIM_FIELDS:
        return "protected_dimension_or_padding"
    if name in REWARD_PATH_FIELDS:
        return "protected_reward_path_milestone"
    if name in CONTROL_COST_FIELDS:
        return "control_cost_or_reward_scale"
    if name in TRANSITION_CONTROL_FIELDS:
        return "transition_or_state_control"
    if name in TERMINAL_SURVIVAL_FIELDS:
        return "terminal_or_survival"
    if name.startswith(TRAINING_OR_PPO_FIELDS_PREFIXES):
        return "training_or_ppo_config"
    if name.startswith("_constrained_"):
        return "sampler_latent"
    if name in {"prior_mlp_hidden_dim", "num_layers", "gp_rff_features", "prior_mlp_activations"}:
        return "model_family_capacity"
    if name in {"noise", "noise_std", "reward_dropout_ratio", "reward_dropout_enabled"}:
        return "noise_or_reward_dropout"
    return "other"


def _rank_auc(xs: list[float | None], labels: list[bool]) -> float | None:
    pairs = [(float(x), bool(y)) for x, y in zip(xs, labels) if x is not None and math.isfinite(float(x))]
    if not pairs:
        return None
    values = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    for val in np.unique(values):
        idx = np.flatnonzero(values == val)
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    rank_sum_pos = float(np.sum(ranks[y]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _safe_stats(values: list[float]) -> dict[str, Any]:
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


def _best_threshold(
    xs: list[float | None],
    labels: list[bool],
    *,
    direction: str,
    recall_min: float,
) -> dict[str, Any] | None:
    pairs = [(float(x), bool(y)) for x, y in zip(xs, labels) if x is not None and math.isfinite(float(x))]
    if not pairs:
        return None
    n_pos = sum(1 for _, y in pairs if y)
    n_total = len(pairs)
    if n_pos <= 0 or n_pos >= n_total:
        return None
    base_rate = float(n_pos / n_total)
    best: tuple[float, float, float, float, float] | None = None
    for threshold in sorted({x for x, _ in pairs}):
        if direction == "low":
            selected = [y for x, y in pairs if x <= threshold]
        else:
            selected = [y for x, y in pairs if x >= threshold]
        if not selected:
            continue
        true_pos = sum(1 for y in selected if y)
        recall = float(true_pos / n_pos)
        if recall < float(recall_min):
            continue
        precision = float(true_pos / len(selected))
        selected_rate = float(len(selected) / n_total)
        lift = float(precision / base_rate) if base_rate > 0.0 else 0.0
        candidate = (lift, precision, recall, -selected_rate, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    lift, precision, recall, neg_selected_rate, threshold = best
    return {
        "direction": direction,
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "base_rate": float(base_rate),
        "precision_lift": float(lift),
        "selected_rate": float(-neg_selected_rate),
        "recall_min": float(recall_min),
    }


def _numeric_relation(rows: list[dict[str, Any]], feature: str, target: str) -> dict[str, Any] | None:
    xs = [_as_float(row["h"].get(feature)) for row in rows]
    labels = [_is_true(row["label"].get(target)) for row in rows]
    valid_xs = [x for x in xs if x is not None and math.isfinite(float(x))]
    if len(valid_xs) < max(8, int(0.8 * len(rows))) or len(set(valid_xs)) < 2:
        return None
    pos = [float(x) for x, y in zip(xs, labels) if y and x is not None]
    neg = [float(x) for x, y in zip(xs, labels) if (not y) and x is not None]
    if len(pos) == 0 or len(neg) == 0:
        return None
    auc = _rank_auc(xs, labels)
    if auc is None:
        return None
    direction = "high" if float(auc) >= 0.5 else "low"
    return {
        "feature": feature,
        "group": _feature_group(feature),
        "target": target,
        "n": len(rows),
        "positive_n": int(sum(labels)),
        "negative_n": int(len(labels) - sum(labels)),
        "valid_n": int(len(valid_xs)),
        "unique_n": int(len(set(valid_xs))),
        "auc": float(auc),
        "direction": direction,
        "strength": float(max(float(auc), 1.0 - float(auc))),
        "positive_stats": _safe_stats(pos),
        "negative_stats": _safe_stats(neg),
        "necessary_threshold_recall90": _best_threshold(
            xs, labels, direction=direction, recall_min=0.90
        ),
    }


def _categorical_relation(rows: list[dict[str, Any]], feature: str, target: str) -> dict[str, Any] | None:
    values = []
    labels = []
    for row in rows:
        value = row["h"].get(feature)
        if isinstance(value, (dict, list)):
            continue
        if value is None:
            continue
        if _as_float(value) is not None:
            continue
        values.append(str(value))
        labels.append(_is_true(row["label"].get(target)))
    if len(values) < max(8, int(0.8 * len(rows))) or len(set(values)) < 2:
        return None
    n_pos = sum(labels)
    if n_pos == 0 or n_pos == len(labels):
        return None
    base_rate = n_pos / len(labels)
    by_value = {}
    for value in sorted(set(values)):
        ys = [y for v, y in zip(values, labels) if v == value]
        by_value[value] = {
            "count": len(ys),
            "positive_n": int(sum(ys)),
            "positive_rate": float(sum(ys) / len(ys)),
            "lift": float((sum(ys) / len(ys)) / base_rate) if base_rate > 0.0 else None,
        }
    return {
        "feature": feature,
        "group": _feature_group(feature),
        "target": target,
        "n": len(values),
        "positive_n": int(n_pos),
        "base_rate": float(base_rate),
        "value_count": len(by_value),
        "by_value": by_value,
    }


def _load_dataset(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    h_list = _load_json(manifest["fixed_h_list_json"])
    label_rows = _read_csv(manifest["label_csv"])
    if len(h_list) != len(label_rows):
        raise RuntimeError(
            f"length mismatch for {manifest_path}: h={len(h_list)} labels={len(label_rows)}"
        )
    rows = [{"h": dict(h), "label": dict(label)} for h, label in zip(h_list, label_rows)]
    return {
        "source_label": manifest.get("source_label"),
        "manifest_json": str(Path(manifest_path).expanduser().resolve()),
        "fixed_h_list_json": str(Path(manifest["fixed_h_list_json"]).expanduser().resolve()),
        "label_csv": str(Path(manifest["label_csv"]).expanduser().resolve()),
        "rows": rows,
    }


def _top_numeric_relations(rows: list[dict[str, Any]], target: str, *, top_k: int) -> list[dict[str, Any]]:
    keys = sorted(set().union(*(set(row["h"].keys()) for row in rows)))
    relations = []
    for key in keys:
        item = _numeric_relation(rows, key, target)
        if item is not None:
            relations.append(item)
    relations.sort(key=lambda item: (float(item["strength"]), item["feature"]), reverse=True)
    return relations[: int(top_k)]


def _specific_relations(rows: list[dict[str, Any]], target: str, features: list[str]) -> list[dict[str, Any]]:
    out = []
    for feature in features:
        item = _numeric_relation(rows, feature, target)
        if item is not None:
            out.append(item)
    return out


def _robust_candidates(dataset_reports: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for ds in dataset_reports:
        for item in ds["targets"][target]["top_numeric"]:
            by_feature.setdefault(item["feature"], []).append(item)
        for item in ds["targets"][target]["specific_numeric"]:
            by_feature.setdefault(item["feature"], []).append(item)
    candidates = []
    for feature, items in by_feature.items():
        source_seen = {item["source_label"] for item in items}
        if not {"full", "q90"}.issubset(source_seen):
            continue
        full_items = [item for item in items if item["source_label"] == "full"]
        q90_items = [item for item in items if item["source_label"] == "q90"]
        full = max(full_items, key=lambda item: float(item["strength"]))
        q90 = max(q90_items, key=lambda item: float(item["strength"]))
        same_direction = full["direction"] == q90["direction"]
        min_strength = min(float(full["strength"]), float(q90["strength"]))
        if same_direction and min_strength >= 0.58:
            candidates.append(
                {
                    "feature": feature,
                    "group": _feature_group(feature),
                    "direction": full["direction"],
                    "full_strength": full["strength"],
                    "q90_strength": q90["strength"],
                    "full_positive_mean": full["positive_stats"].get("mean"),
                    "full_negative_mean": full["negative_stats"].get("mean"),
                    "q90_positive_mean": q90["positive_stats"].get("mean"),
                    "q90_negative_mean": q90["negative_stats"].get("mean"),
                    "full_threshold_recall90": full.get("necessary_threshold_recall90"),
                    "q90_threshold_recall90": q90.get("necessary_threshold_recall90"),
                }
            )
    candidates.sort(key=lambda item: min(float(item["full_strength"]), float(item["q90_strength"])), reverse=True)
    return candidates


def _tertile_table(rows: list[dict[str, Any]], feature: str, targets: tuple[str, ...]) -> dict[str, Any] | None:
    values = [_as_float(row["h"].get(feature)) for row in rows]
    valid = [v for v in values if v is not None and math.isfinite(float(v))]
    if len(valid) < max(8, int(0.8 * len(rows))) or len(set(valid)) < 3:
        return None
    q33, q66 = np.quantile(np.asarray(valid, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    bins = {
        "low": lambda x: x <= q33,
        "mid": lambda x: q33 < x <= q66,
        "high": lambda x: x > q66,
    }
    out = {"feature": feature, "q33": float(q33), "q66": float(q66), "bins": {}}
    for name, predicate in bins.items():
        indices = [idx for idx, value in enumerate(values) if value is not None and predicate(float(value))]
        item: dict[str, Any] = {"n": int(len(indices))}
        for target in targets:
            labels = [_is_true(rows[idx]["label"].get(target)) for idx in indices]
            item[target] = {
                "positive_n": int(sum(labels)),
                "positive_rate": float(sum(labels) / max(1, len(labels))),
            }
        out["bins"][name] = item
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    datasets = [_load_dataset(path) for path in args.manifest_json]
    specific_features = [
        "ctrl_reward_weight",
        "ctrl_reward_enable_prob",
        "reward_scale",
        "state_noise_std",
        "alpha",
        "init_action_std",
        "init_state_std",
        "state_highway_lambda",
        "lengthscale",
        "outputscale",
        "terminal_reset_count_target",
        "reinforce_reward_tanh_bound",
        "prior_mlp_hidden_dim",
        "num_layers",
        "action_dim",
        "obs_dim",
        "state_dim",
    ]
    dataset_reports = []
    for ds in datasets:
        rows = ds["rows"]
        target_reports = {}
        for target in TARGETS:
            top_numeric = _top_numeric_relations(rows, target, top_k=int(args.top_k))
            for item in top_numeric:
                item["source_label"] = ds["source_label"]
            specific_numeric = _specific_relations(rows, target, specific_features)
            for item in specific_numeric:
                item["source_label"] = ds["source_label"]
            target_reports[target] = {
                "positive_n": int(sum(_is_true(row["label"].get(target)) for row in rows)),
                "n": len(rows),
                "top_numeric": top_numeric,
                "specific_numeric": specific_numeric,
            }
        dataset_reports.append(
            {
                "source_label": ds["source_label"],
                "manifest_json": ds["manifest_json"],
                "fixed_h_list_json": ds["fixed_h_list_json"],
                "label_csv": ds["label_csv"],
                "n": len(rows),
                "key_feature_tertiles": {
                    feature: _tertile_table(
                        rows,
                        feature,
                        ("primary_long_horizon_target", "selector_hard_positive"),
                    )
                    for feature in ("ctrl_reward_weight", "state_noise_std", "reward_scale")
                },
                "targets": target_reports,
            }
        )

    robust = {target: _robust_candidates(dataset_reports, target) for target in TARGETS}
    relation_read = {
        "main_candidate_necessary_feature": {
            "feature": "ctrl_reward_weight",
            "direction": "avoid_high",
            "why": (
                "Control-cost pressure is the only mechanistically direct relation that appears "
                "with the same direction in full and q90 for the primary target.  The tertile audit "
                "shows high ctrl_reward_weight suppresses both primary and selector-hard positives. "
                "This explains why state reward identity can be healthy while the optimal policy still "
                "prefers too little action."
            ),
            "not_sufficient": (
                "The recall-90 thresholds select most environments and have low precision, so this is "
                "a necessary tendency / screen precondition, not a standalone milestone fix.  Low and "
                "mid tertiles are similar, so blindly minimizing control cost is not justified."
            ),
        },
        "secondary_unstable_or_indirect_features": [
            {
                "feature": "state_noise_std",
                "read": (
                    "Often positive for primary target, but not stable for selector-hard labels; treat "
                    "as signal/noise difficulty, not a repair knob yet."
                ),
            },
            {
                "feature": "reward_scale",
                "read": (
                    "Higher scale can make action-caused future reward easier to detect, but this is "
                    "not specific to Pendulum control and may confound reward-path milestones."
                ),
            },
            {
                "feature": "alpha/prior_mlp_hidden_dim/terminal_reset_count_target",
                "read": "Directions flip across full/q90 or across targets; not safe as root-cause knobs.",
            },
        ],
        "protected_or_not_for_repair": [
            "action_dim/obs_dim/state_dim are audited but protected by user constraint.",
            "reward_state/action topology fields belong to the prior milestones already protected.",
            "PPO/reinforce settings are not environment generator features for this probe.",
        ],
        "next_probe": (
            "Run signed_action_causal_contraction_screen while stratifying by low/medium/high ctrl_reward_weight. "
            "A real causal relation should show: low cost raises signed future reward gap and closed-loop "
            "contraction without reducing state/reward/done diversity; if not, the control-cost relation is only "
            "a surface correlation."
        ),
    }

    return {
        "analysis_entry": "phase2_pendulum_mode_feature_relation_probe",
        "exploratory_only": True,
        "does_not_sample_or_train": True,
        "does_not_modify_exact_scm": True,
        "does_not_modify_fit_model": True,
        "target_mode": "context_identifiable_signed_closed_loop_contraction",
        "manifest_jsons": [str(Path(p).expanduser().resolve()) for p in args.manifest_json],
        "datasets": dataset_reports,
        "robust_candidate_relations": robust,
        "relation_read": relation_read,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pendulum Mode Feature Relation Probe",
        "",
        "Read-only exploratory report. It relates fixed-list environment `h` features to the purified Pendulum-like target; it does not sample, train, or mutate the generator.",
        "",
        "## Target",
        "",
        "`context_identifiable_signed_closed_loop_contraction`: the model-visible context must reveal a signed closed-loop controller that improves future state reward.",
        "",
        "## Robust Candidate Relations",
        "",
    ]
    robust = report.get("robust_candidate_relations", {}) or {}
    for target in TARGETS:
        lines.append(f"### {target}")
        candidates = robust.get(target, []) or []
        if not candidates:
            lines.append("- no same-direction full/q90 numeric feature reached the robustness threshold")
            continue
        lines.append("| feature | group | dir | full strength | q90 strength | full pos/neg mean | q90 pos/neg mean |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for item in candidates[:12]:
            lines.append(
                (
                    f"| {item['feature']} | {item['group']} | {item['direction']} | "
                    f"{_fmt(item['full_strength'])} | {_fmt(item['q90_strength'])} | "
                    f"{_fmt(item.get('full_positive_mean'))}/{_fmt(item.get('full_negative_mean'))} | "
                    f"{_fmt(item.get('q90_positive_mean'))}/{_fmt(item.get('q90_negative_mean'))} |"
                )
            )
        lines.append("")

    read = report.get("relation_read", {}) or {}
    main = read.get("main_candidate_necessary_feature", {}) or {}
    lines.extend(
        [
            "## Main Read",
            "",
            f"- strongest mechanistic necessary-tendency feature: `{main.get('feature')}` direction `{main.get('direction')}`",
            f"- why: {main.get('why')}",
            f"- caveat: {main.get('not_sufficient')}",
            "",
            "## Why This Is Not The Previous Repair",
            "",
            "- state/reward identity says rewards do not collapse away from state; it does not say action is worth paying for.",
            "- gated reward-path balancing says reward support is healthier; it does not tune the tradeoff between control cost and future state improvement.",
            "- low `ctrl_reward_weight` is closer to Pendulum because official PPO must pay early torque cost to reach a better future basin.",
            "",
            "## Next Falsification",
            "",
            read.get("next_probe", "na"),
            "",
        ]
    )
    lines.append("## Key Feature Tertiles")
    lines.append("")
    lines.append("| list | feature | bin | n | primary rate | hard rate |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for ds in report.get("datasets", []) or []:
        for feature, table in (ds.get("key_feature_tertiles", {}) or {}).items():
            if not table:
                continue
            for bin_name in ("low", "mid", "high"):
                item = (table.get("bins", {}) or {}).get(bin_name, {}) or {}
                lines.append(
                    (
                        f"| {ds.get('source_label')} | {feature} | {bin_name} | {_fmt(item.get('n'))} | "
                        f"{_fmt(_get(item, 'primary_long_horizon_target', 'positive_rate'))} | "
                        f"{_fmt(_get(item, 'selector_hard_positive', 'positive_rate'))} |"
                    )
                )
    lines.append("")
    lines.extend(
        [
            "The actionable reading is to avoid the high-control-cost regime, not to collapse the generator toward zero control cost.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", action="append", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=18)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    json_path = out_dir / "pendulum_mode_feature_relation_probe_report.json"
    md_path = out_dir / "pendulum_mode_feature_relation_probe_summary.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(report, md_path)
    print(json.dumps({"report": str(json_path), "summary": str(md_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
