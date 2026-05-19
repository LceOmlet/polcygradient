#!/usr/bin/env python
"""Extract long-horizon closed-loop control candidates from prior probes.

This is exploratory and read-only with respect to the prior generator,
fit_model, exact SCM defaults, and PPO.  It consumes an existing
Pendulum-like signature report and fixed h-lists, then writes explicit target
subsets and labels for the next selector/smoke probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SIGNATURE_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_prior_pendulum_like_signature_probe_0509/"
    "gated_full_q90_n64_s128_obs4_scales3_timeprofile.json"
)
DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509"

MODE_RE = re.compile(
    r"^(?P<profile>early_|decay_)?obs_linear(?P<neg>_neg)?_(?P<feature>\d+)x(?P<gain>\d+)"
)


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
        return None if not math.isfinite(value) else value
    return value


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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


def _bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key))


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _parse_mode(mode: Any) -> dict[str, Any]:
    if mode is None:
        return {
            "mode": None,
            "family": "none",
            "profile": "none",
            "sign": "none",
            "feature": None,
            "gain": None,
        }
    text = str(mode)
    if text in {"zero", "random", "neg_random", "half_random", "early_random_then_zero"}:
        return {
            "mode": text,
            "family": "open_loop" if text != "zero" else "zero",
            "profile": "early" if text.startswith("early_") else "constant",
            "sign": "random" if "random" in text else "zero",
            "feature": None,
            "gain": None,
        }
    match = MODE_RE.match(text)
    if not match:
        return {
            "mode": text,
            "family": "unknown",
            "profile": "unknown",
            "sign": "unknown",
            "feature": None,
            "gain": None,
        }
    raw_profile = match.group("profile") or ""
    profile = "constant"
    if raw_profile == "early_":
        profile = "early_then_zero"
    elif raw_profile == "decay_":
        profile = "decay"
    return {
        "mode": text,
        "family": "feedback",
        "profile": profile,
        "sign": "neg" if match.group("neg") else "pos",
        "feature": int(match.group("feature")),
        "gain": float(match.group("gain")) / 100.0,
    }


def _labels(row: dict[str, Any]) -> dict[str, bool]:
    nonzero = _bool(row, "nonzero_required_env")
    feedback_open = _bool(row, "feedback_beats_open_loop_env")
    phase = _bool(row, "phase_sensitive_feedback")
    suffix = _bool(row, "suffix_supported")
    time_profile = _bool(row, "time_profile_core")
    strict_decay = _bool(row, "pendulum_like_settle_strict")
    early_carry = _bool(row, "early_only_causal_carryover_core")
    early_strong = _bool(row, "early_only_strong_carryover_core")
    core = nonzero and feedback_open and phase
    return {
        "nonzero_required": nonzero,
        "closed_loop_core": core,
        "suffix_supported": suffix,
        "time_profile_core": time_profile,
        "decay_settle_core": core and strict_decay,
        "sustained_settle_core": core and suffix and time_profile,
        "primary_long_horizon_target": core and suffix and time_profile,
        "early_only_carryover": early_carry,
        "early_only_strong_carryover": early_strong,
        "oracle_global_gap": _bool(row, "oracle_beats_global_by_margin"),
        "global_matches_oracle": _bool(row, "global_feedback_matches_oracle"),
        "selector_hard_positive": core
        and suffix
        and time_profile
        and _bool(row, "oracle_beats_global_by_margin"),
        "selector_easy_positive": core
        and suffix
        and time_profile
        and (not _bool(row, "oracle_beats_global_by_margin")),
    }


def _row_output(row: dict[str, Any], h: dict[str, Any], env_seed: int | None, *, source_label: str) -> dict[str, Any]:
    labels = _labels(row)
    best_feedback = _parse_mode(row.get("best_feedback_mode"))
    best_time = _parse_mode(row.get("best_time_profile_mode"))
    best_prefix = _parse_mode(row.get("best_prefix_mode"))
    out = {
        "source_label": source_label,
        "env_index": int(row["env_index"]),
        "env_seed": env_seed,
        "action_dim": int(h.get("action_dim")),
        "obs_dim": int(h.get("obs_dim")),
        "state_dim": int(h.get("state_dim")),
        "noise_dim": int(h.get("noise_dim")),
        "labels": labels,
        "best_feedback": best_feedback,
        "best_time_profile": best_time,
        "best_prefix": best_prefix,
        "pair_best_sign": row.get("feedback_pair_best_sign"),
        "pair_best_base_mode": row.get("feedback_pair_best_base_mode"),
        "scores": {
            "best_feedback_minus_open_env": _float(row, "best_feedback_minus_open_env"),
            "best_feedback_minus_zero_env": _float(row, "best_feedback_minus_zero_env"),
            "feedback_pair_phase_gap_env": _float(row, "feedback_pair_phase_gap_env"),
            "feedback_suffix_gain_over_open_env": _float(row, "feedback_suffix_gain_over_open_env"),
            "feedback_suffix_gain_over_zero_env": _float(row, "feedback_suffix_gain_over_zero_env"),
            "best_time_profile_minus_open_env": (
                _float(row, "best_time_profile_env_return")
                - _float(row, "best_open_loop_env_return")
                if _float(row, "best_time_profile_env_return") is not None
                and _float(row, "best_open_loop_env_return") is not None
                else None
            ),
            "early_only_suffix_gain_over_open_env": _float(row, "early_only_suffix_gain_over_open_env"),
            "oracle_minus_global_feedback_env": _float(row, "oracle_minus_global_feedback_env"),
            "best_prefix_minus_open_loop_env": _float(row, "best_prefix_minus_open_loop_env"),
            "best_prefix_minus_global_prefix_env": _float(row, "best_prefix_minus_global_prefix_env"),
        },
        "generation_knobs": {
            "alpha": h.get("alpha"),
            "lengthscale": h.get("lengthscale"),
            "reward_scale": h.get("reward_scale"),
            "reward_dropout_ratio": h.get("reward_dropout_ratio"),
            "ctrl_reward_weight": h.get("ctrl_reward_weight"),
            "init_state_std": h.get("init_state_std"),
            "init_action_std": h.get("init_action_std"),
            "state_noise_std": h.get("state_noise_std"),
            "terminal_reset_count_target": h.get("terminal_reset_count_target"),
        },
    }
    return out


def _counter(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        counter[str(value)] += 1
    return dict(counter)


def _score_stats(rows: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get("scores", {}).get(score_name)
        if value is not None:
            values.append(float(value))
    return _stats(values)


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": int(len(rows)),
        "source_counts": _counter(rows, ("source_label",)),
        "feedback_mode_counts": _counter(rows, ("best_feedback", "mode")),
        "feedback_profile_counts": _counter(rows, ("best_feedback", "profile")),
        "feedback_sign_counts": _counter(rows, ("best_feedback", "sign")),
        "feedback_feature_counts": _counter(rows, ("best_feedback", "feature")),
        "feedback_gain_counts": _counter(rows, ("best_feedback", "gain")),
        "time_profile_mode_counts": _counter(rows, ("best_time_profile", "mode")),
        "time_profile_profile_counts": _counter(rows, ("best_time_profile", "profile")),
        "pair_sign_counts": _counter(rows, ("pair_best_sign",)),
        "action_dim": _stats([float(row["action_dim"]) for row in rows]),
        "obs_dim": _stats([float(row["obs_dim"]) for row in rows]),
        "state_dim": _stats([float(row["state_dim"]) for row in rows]),
        "oracle_minus_global_feedback_env": _score_stats(rows, "oracle_minus_global_feedback_env"),
        "best_feedback_minus_open_env": _score_stats(rows, "best_feedback_minus_open_env"),
        "feedback_pair_phase_gap_env": _score_stats(rows, "feedback_pair_phase_gap_env"),
        "feedback_suffix_gain_over_open_env": _score_stats(rows, "feedback_suffix_gain_over_open_env"),
        "best_prefix_minus_open_loop_env": _score_stats(rows, "best_prefix_minus_open_loop_env"),
    }


def _deficit_report(selected: list[dict[str, Any]], total: int) -> dict[str, Any]:
    ratio = float(len(selected) / max(1, total))
    sign_counts = _counter(selected, ("best_feedback", "sign"))
    feature_counts = _counter(selected, ("best_feedback", "feature"))
    gain_counts = _counter(selected, ("best_feedback", "gain"))
    profile_counts = _counter(selected, ("best_feedback", "profile"))

    def _balance_status(counts: dict[str, int]) -> dict[str, Any]:
        nonzero = {k: int(v) for k, v in counts.items() if int(v) > 0 and str(k) != "None"}
        n = int(sum(nonzero.values()))
        if n <= 0:
            return {"status": "empty", "max_share": None, "dominant": None, "counts": counts}
        dominant, dominant_n = max(nonzero.items(), key=lambda item: int(item[1]))
        max_share = float(dominant_n / max(1, n))
        if len(nonzero) < 2:
            status = "collapsed"
        elif max_share > 0.75:
            status = "dominated"
        else:
            status = "ok"
        return {
            "status": status,
            "max_share": max_share,
            "dominant": dominant,
            "counts": counts,
        }

    return {
        "observed_ratio": ratio,
        "suggested_screening_ratio_before_fit_longrun": "0.20_to_0.30",
        "ratio_status": (
            "within_initial_target"
            if 0.20 <= ratio <= 0.35
            else ("too_sparse_for_reliable_fit_signal" if ratio < 0.20 else "may_overdominate_distribution")
        ),
        "balance_status": {
            "sign": _balance_status(sign_counts),
            "feature": _balance_status(feature_counts),
            "gain": _balance_status(gain_counts),
            "profile": _balance_status(profile_counts),
        },
        "note": (
            "This is a screening/quota diagnostic. It is not evidence that changing "
            "parameter samplers is safe; behavior labels remain authoritative."
        ),
    }


def _process_list(item: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    label = str(item["label"])
    short_label = "q90" if "_q90_" in label else "full"
    h_path = Path(item["fixed_h_list_json"]).expanduser().resolve()
    seed_path = Path(item["fixed_env_seed_json"]).expanduser().resolve()
    h_list = _load_json(h_path)
    seeds = _load_json(seed_path) if seed_path.exists() else [None] * len(h_list)
    per_env = item["classification"]["per_env"]
    if not (len(h_list) == len(per_env) == len(seeds)):
        raise RuntimeError(f"{label}: length mismatch h={len(h_list)} per_env={len(per_env)} seeds={len(seeds)}")

    rows = [
        _row_output(row, h, int(seed) if seed is not None else None, source_label=short_label)
        for row, h, seed in zip(per_env, h_list, seeds)
    ]
    subsets = {
        "closed_loop_core": [row for row in rows if row["labels"]["closed_loop_core"]],
        "primary_long_horizon_target": [
            row for row in rows if row["labels"]["primary_long_horizon_target"]
        ],
        "decay_settle_core": [row for row in rows if row["labels"]["decay_settle_core"]],
        "early_only_carryover": [row for row in rows if row["labels"]["early_only_carryover"]],
        "selector_hard_positive": [row for row in rows if row["labels"]["selector_hard_positive"]],
        "selector_easy_positive": [row for row in rows if row["labels"]["selector_easy_positive"]],
    }

    list_dir = out_dir / short_label
    list_dir.mkdir(parents=True, exist_ok=True)
    csv_path = list_dir / "long_horizon_control_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_label",
            "env_index",
            "env_seed",
            "action_dim",
            "obs_dim",
            "state_dim",
            "closed_loop_core",
            "primary_long_horizon_target",
            "decay_settle_core",
            "early_only_carryover",
            "selector_hard_positive",
            "selector_easy_positive",
            "best_feedback_mode",
            "best_feedback_profile",
            "best_feedback_sign",
            "best_feedback_feature",
            "best_feedback_gain",
            "pair_best_sign",
            "best_time_profile_mode",
            "oracle_minus_global_feedback_env",
            "best_feedback_minus_open_env",
            "feedback_suffix_gain_over_open_env",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_label": row["source_label"],
                    "env_index": row["env_index"],
                    "env_seed": row["env_seed"],
                    "action_dim": row["action_dim"],
                    "obs_dim": row["obs_dim"],
                    "state_dim": row["state_dim"],
                    "closed_loop_core": row["labels"]["closed_loop_core"],
                    "primary_long_horizon_target": row["labels"]["primary_long_horizon_target"],
                    "decay_settle_core": row["labels"]["decay_settle_core"],
                    "early_only_carryover": row["labels"]["early_only_carryover"],
                    "selector_hard_positive": row["labels"]["selector_hard_positive"],
                    "selector_easy_positive": row["labels"]["selector_easy_positive"],
                    "best_feedback_mode": row["best_feedback"]["mode"],
                    "best_feedback_profile": row["best_feedback"]["profile"],
                    "best_feedback_sign": row["best_feedback"]["sign"],
                    "best_feedback_feature": row["best_feedback"]["feature"],
                    "best_feedback_gain": row["best_feedback"]["gain"],
                    "pair_best_sign": row["pair_best_sign"],
                    "best_time_profile_mode": row["best_time_profile"]["mode"],
                    "oracle_minus_global_feedback_env": row["scores"]["oracle_minus_global_feedback_env"],
                    "best_feedback_minus_open_env": row["scores"]["best_feedback_minus_open_env"],
                    "feedback_suffix_gain_over_open_env": row["scores"]["feedback_suffix_gain_over_open_env"],
                }
            )

    target_indices = [int(row["env_index"]) for row in subsets["primary_long_horizon_target"]]
    target_h = [h_list[idx] for idx in target_indices]
    target_seeds = [seeds[idx] for idx in target_indices]
    target_h_path = list_dir / "primary_long_horizon_target_h_list.json"
    target_seed_path = list_dir / "primary_long_horizon_target_env_seeds.json"
    target_h_path.write_text(json.dumps(_json_safe(target_h), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target_seed_path.write_text(json.dumps(_json_safe(target_seeds), indent=2) + "\n", encoding="utf-8")

    selector_manifest = {
        "source_label": short_label,
        "fixed_h_list_json": str(h_path),
        "fixed_env_seed_json": str(seed_path),
        "label_csv": str(csv_path),
        "primary_target_h_list_json": str(target_h_path),
        "primary_target_env_seed_json": str(target_seed_path),
        "selector_target": {
            "classification": "best_feedback mode/sign/feature/gain and time-profile family",
            "positive_mask": "primary_long_horizon_target",
            "hard_positive_mask": "selector_hard_positive",
            "negative_controls": [
                "closed_loop_core_without_time_profile_or_suffix",
                "open_loop_best_or_zero_best",
                "early_only_carryover_without_full_feedback_settle",
            ],
        },
    }
    manifest_path = list_dir / "selector_probe_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(selector_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "label": label,
        "short_label": short_label,
        "n_envs": int(len(rows)),
        "label_csv": str(csv_path),
        "primary_target_h_list_json": str(target_h_path),
        "primary_target_env_seed_json": str(target_seed_path),
        "selector_probe_manifest_json": str(manifest_path),
        "subsets": {name: _subset_summary(values) for name, values in subsets.items()},
        "primary_target_deficit": _deficit_report(subsets["primary_long_horizon_target"], len(rows)),
        "definition": {
            "primary_long_horizon_target": (
                "nonzero_required_env AND feedback_beats_open_loop_env AND "
                "phase_sensitive_feedback AND suffix_supported AND time_profile_core"
            ),
            "why_early_only_is_secondary": (
                "Gym Pendulum official early-then-zero improves over fit but remains far below "
                "full official control, so early-only carryover is not the primary target."
            ),
            "why_global_oracle_gap_is_not_positive_label": (
                "A large oracle-global gap marks context-identifiability risk. It should drive "
                "selector testing, not direct enrichment by itself."
            ),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    signature = _load_json(args.signature_json)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    list_reports = [_process_list(item, out_dir) for item in signature["lists"]]
    return {
        "analysis_entry": "phase2_long_horizon_control_mode_extract",
        "exploratory_only": True,
        "does_not_modify_exact_scm": True,
        "does_not_modify_fit_model": True,
        "does_not_train_or_sample": True,
        "input_signature_json": str(Path(args.signature_json).expanduser().resolve()),
        "mode_definition": {
            "name": "context_identifiable_closed_loop_settle_feedback",
            "primary_behavioral_predicate": (
                "nonzero action necessary, state-conditioned feedback beats open-loop, "
                "phase/sign matters, suffix benefit exists, and a time-profile/settle "
                "controller beats open-loop"
            ),
            "selector_question": (
                "Can context/frozen hidden choose the right feedback mode/sign/feature/gain "
                "on heldout envs, rather than relying on per-env oracle labels?"
            ),
            "generation_control_interpretation": (
                "Use the predicate as a post-sampling quota/screen first. Only if selector and "
                "short PPO smoke pass should generator parameters be adjusted."
            ),
        },
        "lists": list_reports,
        "next_step": {
            "recommended": "context_to_controller_selector_probe",
            "inputs": [item["selector_probe_manifest_json"] for item in list_reports],
            "acceptance": [
                "heldout selector rollout beats global feedback and open-loop controls",
                "selector recovers useful sign/feature/gain without collapsing to one mode",
                "candidate fixed-list short PPO improves raw env return and late stability",
            ],
        },
    }


def _write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Long-Horizon Control Mode Extract",
        "",
        "Exploratory only. It extracts labels/subsets from existing prior signature reports; no sampling or training.",
        "",
        "## Primary Mode",
        "",
        report["mode_definition"]["primary_behavioral_predicate"],
        "",
        "Early-only carryover is secondary, not the primary target, because Gym Pendulum official early-then-zero remains insufficient.",
        "",
        "## Lists",
    ]
    for item in report["lists"]:
        primary = item["subsets"]["primary_long_horizon_target"]
        hard = item["subsets"]["selector_hard_positive"]
        easy = item["subsets"]["selector_easy_positive"]
        lines.extend(
            [
                f"### {item['short_label']}",
                f"- n_envs: {item['n_envs']}",
                f"- primary target n/ratio: {primary['n']} / {primary['n'] / max(1, item['n_envs']):.6f}",
                f"- selector hard positive n: {hard['n']}",
                f"- selector easy positive n: {easy['n']}",
                f"- ratio status: {item['primary_target_deficit']['ratio_status']}",
                f"- balance status: {item['primary_target_deficit']['balance_status']}",
                f"- label csv: `{item['label_csv']}`",
                f"- target h list: `{item['primary_target_h_list_json']}`",
                f"- selector manifest: `{item['selector_probe_manifest_json']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Next",
            "",
            "Run a context-to-controller selector probe before changing generator parameters or starting a fit_model long run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-json", default=DEFAULT_SIGNATURE_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_report(args)
    out_dir = Path(args.output_dir).expanduser().resolve()
    json_path = out_dir / "long_horizon_control_mode_extract_report.json"
    md_path = out_dir / "long_horizon_control_mode_extract_summary.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(report, md_path)
    print(json.dumps({"report": str(json_path), "summary": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
