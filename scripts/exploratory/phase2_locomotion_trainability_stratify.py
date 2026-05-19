#!/usr/bin/env python
"""Stratify short PPO trainability by GPU0 locomotion temporal witnesses.

Exploratory/read-only analysis. It consumes existing runner sidecars and mode
detector artifacts. It does not sample environments, train PPO, or mutate
milestone/default generator code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_locomotion_trainability_stratification_gpu0_0510"
DEFAULT_LOCOMOTION_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_full_temporal_detector_gpu0_0510/"
    "locomotion_full_temporal_per_env.csv"
)
DEFAULT_ACTION_MODE_JSON = (
    "/home/chen/RLPFN/artifacts/phase2_prior_action_mode_coverage_probe_0508/"
    "gated_full_q90_n64_s256_v2.json"
)
DEFAULT_FULL_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_trainability_full_sidecar_s512_u4_e4_advnorm_gpu0_0510"
)
DEFAULT_Q90_RUN_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_locomotion_trainability_q90_sidecar_s512_u4_e4_advnorm_gpu0_0510"
)
DEFAULT_FULL_LONG_HORIZON_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "full/long_horizon_control_labels.csv"
)
DEFAULT_Q90_LONG_HORIZON_CSV = (
    "/home/chen/RLPFN/artifacts/phase2_long_horizon_control_mode_extract_0509/"
    "q90/long_horizon_control_labels.csv"
)


SCOPE_TO_RUN = {
    "phase2_exploratory_reward_group_balance_gated_full_fixedlist_n64_s512_u4_e4_advnorm_0506": {
        "short": "full",
        "run_dir": DEFAULT_FULL_RUN_DIR,
        "long_horizon_csv": DEFAULT_FULL_LONG_HORIZON_CSV,
    },
    "phase2_exploratory_reward_group_balance_gated_q90_fixedlist_n64_s512_u4_e4_advnorm_0506": {
        "short": "q90",
        "run_dir": DEFAULT_Q90_RUN_DIR,
        "long_horizon_csv": DEFAULT_Q90_LONG_HORIZON_CSV,
    },
}


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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _delta(a: Any, b: Any) -> float | None:
    x = _f(a)
    y = _f(b)
    if x is None or y is None:
        return None
    return float(y - x)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if _f(v) is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
    }


def _sidecar_path(run_dir: str | Path, update_idx: int) -> Path:
    return Path(run_dir).expanduser().resolve() / "semantic_sidecars" / f"prior_update_{int(update_idx):06d}_envs.jsonl"


def _load_action_mode_index(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = _load_json(path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for item in payload.get("lists", []):
        label = str(item.get("label"))
        for row in item.get("classification", {}).get("per_env", []):
            out[(label, int(row["env_index"]))] = row
    return out


def _load_locomotion_index(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_csv(path):
        out[(str(row["scope"]), int(row["env_index"]))] = row
    return out


def _load_long_horizon_index(path: str | Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in _load_csv(path):
        out[int(row["env_index"])] = row
    return out


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": int(len(rows)),
        "raw_reward_sum_delta": _stats([r.get("raw_reward_sum_delta") for r in rows]),
        "reward_env_sum_delta": _stats([r.get("reward_env_sum_delta") for r in rows]),
        "training_reward_sum_delta": _stats([r.get("training_reward_sum_delta") for r in rows]),
        "done_count_delta": _stats([r.get("done_count_delta") for r in rows]),
        "first_done_step_delta": _stats([r.get("first_done_step_delta") for r in rows]),
        "action_value_absmean_delta": _stats([r.get("action_value_absmean_delta") for r in rows]),
        "action_saturation_delta": _stats([r.get("action_saturation_delta") for r in rows]),
        "next_state_step_l2_delta_mean_delta": _stats(
            [r.get("next_state_step_l2_delta_mean_delta") for r in rows]
        ),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row.get(key)) for row in rows]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    locomotion = _load_locomotion_index(args.locomotion_csv)
    action_modes = _load_action_mode_index(args.action_mode_json)
    per_env: list[dict[str, Any]] = []
    scope_reports: dict[str, Any] = {}

    for scope, cfg in SCOPE_TO_RUN.items():
        run_dir = Path(args.full_run_dir if cfg["short"] == "full" else args.q90_run_dir).expanduser().resolve()
        long_horizon = _load_long_horizon_index(
            args.full_long_horizon_csv if cfg["short"] == "full" else args.q90_long_horizon_csv
        )
        first_rows = {int(row["env_idx"]): row for row in _load_jsonl(_sidecar_path(run_dir, args.first_update))}
        last_rows = {int(row["env_idx"]): row for row in _load_jsonl(_sidecar_path(run_dir, args.last_update))}
        if set(first_rows) != set(last_rows):
            raise RuntimeError(f"{scope}: first/last sidecar env mismatch")
        rows: list[dict[str, Any]] = []
        for env_idx in sorted(first_rows):
            first = first_rows[env_idx]
            last = last_rows[env_idx]
            loc = locomotion.get((scope, env_idx), {})
            action = action_modes.get((scope, env_idx), {})
            lh = long_horizon.get(env_idx, {})
            row = {
                "scope": scope,
                "scope_short": cfg["short"],
                "env_idx": int(env_idx),
                "env_group_env_seed": first.get("env_group_env_seed"),
                "locomotion_witness": _bool(loc.get("accepted")),
                "locomotion_mode": loc.get("mode"),
                "locomotion_env_reward_gain_vs_zero": _f(loc.get("env_reward_gain_vs_zero")),
                "locomotion_state_path_gain_vs_zero": _f(loc.get("state_path_gain_vs_zero")),
                "low_energy_not_ctrl_only": bool(action.get("low_energy_not_ctrl_only", False)),
                "sign_or_phase_sensitive": bool(action.get("sign_or_phase_sensitive_present", False)),
                "state_conditioned_energy_injection": bool(
                    action.get("state_conditioned_energy_injection_witness_present", False)
                ),
                "state_conditioned_env_reward": bool(
                    action.get("state_conditioned_env_reward_witness_present", False)
                ),
                "pendulum_like_primary_long_horizon": _bool(lh.get("primary_long_horizon_target")),
                "pendulum_like_closed_loop_core": _bool(lh.get("closed_loop_core")),
                "pendulum_like_selector_hard_positive": _bool(lh.get("selector_hard_positive")),
                "pendulum_like_best_feedback_profile": lh.get("best_feedback_profile"),
                "pendulum_like_best_feedback_gain": _f(lh.get("best_feedback_gain")),
                # Not a formal terminal-sensitive detector. This is a same-run terminal activity proxy.
                "terminal_metadata_enabled": bool(first.get("terminal_reset_enabled", False)),
                "terminal_activity_first_done": first.get("first_done_step") is not None,
                "terminal_activity_done_count_first": _f(first.get("done_count")),
                "raw_reward_sum_first": _f(first.get("raw_reward_sum")),
                "raw_reward_sum_last": _f(last.get("raw_reward_sum")),
                "raw_reward_sum_delta": _delta(first.get("raw_reward_sum"), last.get("raw_reward_sum")),
                "reward_env_sum_delta": _delta(first.get("reward_env_sum"), last.get("reward_env_sum")),
                "training_reward_sum_delta": _delta(
                    first.get("training_reward_sum"), last.get("training_reward_sum")
                ),
                "reward_ctrl_sum_delta": _delta(first.get("reward_ctrl_sum"), last.get("reward_ctrl_sum")),
                "done_count_delta": _delta(first.get("done_count"), last.get("done_count")),
                "first_done_step_delta": _delta(first.get("first_done_step"), last.get("first_done_step")),
                "action_value_absmean_delta": _delta(
                    first.get("action_value_absmean"), last.get("action_value_absmean")
                ),
                "action_saturation_delta": _delta(
                    first.get("raw_policy_action_unit_clip_saturation_fraction"),
                    last.get("raw_policy_action_unit_clip_saturation_fraction"),
                ),
                "next_state_step_l2_delta_mean_delta": _delta(
                    first.get("next_state_step_l2_delta_mean"),
                    last.get("next_state_step_l2_delta_mean"),
                ),
            }
            rows.append(row)
            per_env.append(row)
        witness = [r for r in rows if r["locomotion_witness"]]
        non = [r for r in rows if not r["locomotion_witness"]]
        scope_reports[scope] = {
            "scope_short": cfg["short"],
            "run_dir": str(run_dir),
            "n_envs": int(len(rows)),
            "locomotion_witness_count": int(len(witness)),
            "locomotion_nonwitness_count": int(len(non)),
            "witness": _summarize_group(witness),
            "nonwitness": _summarize_group(non),
            "all": _summarize_group(rows),
            "mode_rates": {
                "low_energy_not_ctrl_only": {
                    "all": _rate(rows, "low_energy_not_ctrl_only"),
                    "witness": _rate(witness, "low_energy_not_ctrl_only"),
                    "nonwitness": _rate(non, "low_energy_not_ctrl_only"),
                },
                "pendulum_like_primary_long_horizon": {
                    "all": _rate(rows, "pendulum_like_primary_long_horizon"),
                    "witness": _rate(witness, "pendulum_like_primary_long_horizon"),
                    "nonwitness": _rate(non, "pendulum_like_primary_long_horizon"),
                },
                "pendulum_like_selector_hard_positive": {
                    "all": _rate(rows, "pendulum_like_selector_hard_positive"),
                    "witness": _rate(witness, "pendulum_like_selector_hard_positive"),
                    "nonwitness": _rate(non, "pendulum_like_selector_hard_positive"),
                },
                "terminal_metadata_enabled": {
                    "all": _rate(rows, "terminal_metadata_enabled"),
                    "witness": _rate(witness, "terminal_metadata_enabled"),
                    "nonwitness": _rate(non, "terminal_metadata_enabled"),
                },
                "terminal_activity_first_done": {
                    "all": _rate(rows, "terminal_activity_first_done"),
                    "witness": _rate(witness, "terminal_activity_first_done"),
                    "nonwitness": _rate(non, "terminal_activity_first_done"),
                },
                "state_conditioned_energy_injection": {
                    "all": _rate(rows, "state_conditioned_energy_injection"),
                    "witness": _rate(witness, "state_conditioned_energy_injection"),
                    "nonwitness": _rate(non, "state_conditioned_energy_injection"),
                },
            },
            "witness_env_indices": [int(r["env_idx"]) for r in witness],
            "top_witness_regressed": sorted(
                witness,
                key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0),
            )[:10],
            "top_nonwitness_improved": sorted(
                non,
                key=lambda r: float(r.get("raw_reward_sum_delta") or 0.0),
                reverse=True,
            )[:10],
        }

    report = {
        "analysis_entry": "phase2_locomotion_trainability_stratification_gpu0",
        "contract": {
            "exploratory_only": True,
            "reads_existing_sidecars_and_mode_labels": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_environments": True,
            "does_not_train": True,
            "terminal_sensitive_label_status": (
                "no same-list action-sensitive terminal detector; terminal fields here are activity proxies only"
            ),
        },
        "first_update": int(args.first_update),
        "last_update": int(args.last_update),
        "sources": {
            "locomotion_csv": str(Path(args.locomotion_csv).expanduser().resolve()),
            "action_mode_json": str(Path(args.action_mode_json).expanduser().resolve()),
            "full_run_dir": str(Path(args.full_run_dir).expanduser().resolve()),
            "q90_run_dir": str(Path(args.q90_run_dir).expanduser().resolve()),
            "full_long_horizon_csv": str(Path(args.full_long_horizon_csv).expanduser().resolve()),
            "q90_long_horizon_csv": str(Path(args.q90_long_horizon_csv).expanduser().resolve()),
        },
        "scopes": scope_reports,
        "per_env": per_env,
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "locomotion_trainability_stratification_report.json"
    csv_path = out_dir / "locomotion_trainability_per_env.csv"
    md_path = out_dir / "locomotion_trainability_stratification.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if per_env:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_env[0].keys()))
            writer.writeheader()
            writer.writerows(_json_safe(per_env))
    lines = [
        "# Locomotion Trainability Stratification GPU0",
        "",
        "Read-only summary from GPU0 locomotion witnesses and sidecar short PPO runs.",
        "",
    ]
    for scope, item in scope_reports.items():
        lines.append(f"## {item['scope_short']}")
        lines.append("")
        lines.append(
            f"- witnesses: {item['locomotion_witness_count']}/{item['n_envs']}"
        )
        for group in ("witness", "nonwitness", "all"):
            stats = item[group]["raw_reward_sum_delta"]
            lines.append(
                f"- {group}: raw_delta_mean={stats.get('mean')}, "
                f"positive_fraction={stats.get('positive_fraction')}, "
                f"q10={stats.get('q10')}, q50={stats.get('q50')}, q90={stats.get('q90')}"
            )
        lines.append("")
        lines.append("| mode/proxy | all | witness | nonwitness |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key, rates in item["mode_rates"].items():
            lines.append(
                f"| {key} | {rates['all']:.3f} | {rates['witness']:.3f} | {rates['nonwitness']:.3f} |"
            )
        lines.append("")
    lines.append("## Limitation")
    lines.append("")
    lines.append(
        "Terminal-sensitive mode is not formally available on the same full/q90 fixed lists. "
        "The terminal rows above are terminal activity proxies, not an action-sensitive terminal detector."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locomotion-csv", default=DEFAULT_LOCOMOTION_CSV)
    parser.add_argument("--action-mode-json", default=DEFAULT_ACTION_MODE_JSON)
    parser.add_argument("--full-run-dir", default=DEFAULT_FULL_RUN_DIR)
    parser.add_argument("--q90-run-dir", default=DEFAULT_Q90_RUN_DIR)
    parser.add_argument("--full-long-horizon-csv", default=DEFAULT_FULL_LONG_HORIZON_CSV)
    parser.add_argument("--q90-long-horizon-csv", default=DEFAULT_Q90_LONG_HORIZON_CSV)
    parser.add_argument("--first-update", type=int, default=0)
    parser.add_argument("--last-update", type=int, default=3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": args.output_dir,
                    "scopes": {
                        k: {
                            "witnesses": v["locomotion_witness_count"],
                            "witness_raw_delta_mean": v["witness"]["raw_reward_sum_delta"].get("mean"),
                            "nonwitness_raw_delta_mean": v["nonwitness"]["raw_reward_sum_delta"].get("mean"),
                        }
                        for k, v in report["scopes"].items()
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
