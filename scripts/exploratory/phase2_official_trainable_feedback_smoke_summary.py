#!/usr/bin/env python
"""Summarize official-trainable feedback fixed-list smoke runs.

Exploratory only. Reads trusted pack runner outputs and checks short-training
optimisability plus basic state/reward/done diversity guards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = "/home/chen/RLPFN/artifacts/phase2_official_trainable_feedback_cross_seed_smoke_summary_0509"


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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _stat(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray(_finite(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "q50": float(np.quantile(arr, 0.5)),
        "max": float(np.max(arr)),
    }


def _delta(first: Any, last: Any) -> float | None:
    try:
        f = float(first)
        l = float(last)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(f) and math.isfinite(l)):
        return None
    return float(l - f)


def _summarize_run(name: str, run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    summary = _load_json(root / "summary.json")
    rows = _load_jsonl(root / "prior_progress.jsonl")
    out: dict[str, Any] = {
        "name": name,
        "run_dir": str(root),
        "updates": int(len(rows)),
        "summary_hard_pass": bool(summary.get("hard_pass")),
        "row_hard_pass": bool(rows) and all((row.get("update") or {}).get("exception") is None for row in rows),
    }
    if not rows:
        out["hard_pass"] = False
        return out

    series = []
    for idx, row in enumerate(rows):
        stats = (row.get("pack_summary") or {}).get("stats") or {}
        reward_component = row.get("reward_component_summary") or {}
        collector = row.get("collector") or {}
        logger = (row.get("update") or {}).get("logger") or {}
        series.append(
            {
                "idx": int(idx),
                "raw_reward_return_mean": stats.get("raw_reward_return_mean"),
                "normalized_reward_return_mean": stats.get("reward_return_mean"),
                "full_env_return_mean": reward_component.get("full_reward_env_return_mean"),
                "ep_rew_mean": reward_component.get("ep_rew_mean"),
                "full_ep_len_mean": reward_component.get("full_ep_len_mean"),
                "done_count": stats.get("done_count"),
                "raw_reward_std": stats.get("raw_reward_std"),
                "reward_std": stats.get("reward_std"),
                "action_absmean": stats.get("action_absmean"),
                "approx_kl": logger.get("train/approx_kl"),
                "clip_fraction": logger.get("train/clip_fraction"),
                "action_dims": collector.get("action_dims") or [],
                "obs_dims": collector.get("obs_dims") or [],
                "state_dims": collector.get("state_dims") or [],
            }
        )
    first = series[0]
    last = series[-1]
    action_dims = list(last.get("action_dims") or [])
    obs_dims = list(last.get("obs_dims") or [])
    state_dims = list(last.get("state_dims") or [])
    diversity_guard = {
        "done_count_min_positive": min(int(x) for x in _finite([s["done_count"] for s in series])) > 0,
        "raw_reward_std_min": min(_finite([s["raw_reward_std"] for s in series])),
        "reward_std_min": min(_finite([s["reward_std"] for s in series])),
        "action_absmean_min": min(_finite([s["action_absmean"] for s in series])),
        "unique_action_dims": len(set(int(v) for v in action_dims)),
        "unique_obs_dims": len(set(int(v) for v in obs_dims)),
        "unique_state_dims": len(set(int(v) for v in state_dims)),
    }
    diversity_guard["pass"] = (
        bool(diversity_guard["done_count_min_positive"])
        and float(diversity_guard["raw_reward_std_min"]) > 1e-6
        and float(diversity_guard["reward_std_min"]) > 1e-8
        and float(diversity_guard["action_absmean_min"]) > 1e-6
        and int(diversity_guard["unique_action_dims"]) >= min(2, len(action_dims))
        and int(diversity_guard["unique_obs_dims"]) >= min(2, len(obs_dims))
        and int(diversity_guard["unique_state_dims"]) >= min(2, len(state_dims))
    )
    out.update(
        {
            "n_envs": int((summary.get("contract") or {}).get("n_envs", 0)),
            "contract": summary.get("contract") or {},
            "series": series,
            "deltas": {
                "raw_reward_return_mean": _delta(first["raw_reward_return_mean"], last["raw_reward_return_mean"]),
                "normalized_reward_return_mean": _delta(
                    first["normalized_reward_return_mean"], last["normalized_reward_return_mean"]
                ),
                "full_env_return_mean": _delta(first["full_env_return_mean"], last["full_env_return_mean"]),
                "ep_rew_mean": _delta(first["ep_rew_mean"], last["ep_rew_mean"]),
                "full_ep_len_mean": _delta(first["full_ep_len_mean"], last["full_ep_len_mean"]),
                "done_count": _delta(first["done_count"], last["done_count"]),
            },
            "first": first,
            "last": last,
            "diversity_guard": diversity_guard,
        }
    )
    out["hard_pass"] = bool(out["summary_hard_pass"]) and bool(out["row_hard_pass"]) and bool(diversity_guard["pass"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    runs = [_summarize_run(name, path) for name, path in args.run]
    by_name = {str(item["name"]): item for item in runs}
    trainable = next((item for item in runs if "trainable" in str(item["name"]) and "lost" not in str(item["name"])), None)
    trainlost = next((item for item in runs if "lost" in str(item["name"])), None)
    predictive = None
    if trainable is not None and trainlost is not None:
        predictive = {
            "raw_delta_margin": _delta(
                (trainlost.get("deltas") or {}).get("raw_reward_return_mean"),
                (trainable.get("deltas") or {}).get("raw_reward_return_mean"),
            ),
            "full_env_delta_margin": _delta(
                (trainlost.get("deltas") or {}).get("full_env_return_mean"),
                (trainable.get("deltas") or {}).get("full_env_return_mean"),
            ),
            "ep_rew_delta_margin": _delta(
                (trainlost.get("deltas") or {}).get("ep_rew_mean"),
                (trainable.get("deltas") or {}).get("ep_rew_mean"),
            ),
        }
    report = {
        "analysis_entry": "phase2_official_trainable_feedback_smoke_summary",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "trusted_pack_runner_outputs_only": True,
        },
        "runs": by_name,
        "predictive_check": predictive,
        "hard_pass": all(bool(item.get("hard_pass")) for item in runs),
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "official_trainable_feedback_smoke_summary_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Official-Trainable Feedback Smoke Summary",
        "",
        f"hard_pass: {report['hard_pass']}",
        "",
        "| group | n_envs | raw delta | normalized delta | full env delta | ep_rew delta | ep_len delta | done delta | diversity pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in runs:
        d = item.get("deltas") or {}
        lines.append(
            f"| {item['name']} | {item.get('n_envs')} | {d.get('raw_reward_return_mean')} | "
            f"{d.get('normalized_reward_return_mean')} | {d.get('full_env_return_mean')} | "
            f"{d.get('ep_rew_mean')} | {d.get('full_ep_len_mean')} | {d.get('done_count')} | "
            f"{(item.get('diversity_guard') or {}).get('pass')} |"
        )
    if predictive is not None:
        lines.extend(
            [
                "",
                "## Predictive Check",
                "",
                f"- raw delta margin, trainable minus train-lost: {predictive.get('raw_delta_margin')}",
                f"- full-env delta margin, trainable minus train-lost: {predictive.get('full_env_delta_margin')}",
                f"- ep_rew delta margin, trainable minus train-lost: {predictive.get('ep_rew_delta_margin')}",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a short fixed-list smoke. It can support or falsify whether the official-trainable label predicts short optimisability, but it is not a milestone acceptance test by itself.",
        ]
    )
    md_path = out_dir / "official_trainable_feedback_smoke_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(md_path), "report": str(report_path), "hard_pass": report["hard_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
