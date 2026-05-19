#!/usr/bin/env python3
"""Build the formal full/q90 multi-seed U4 opt-parity report for M4 survival labels.

This script is a read-only gate over already-produced runner artifacts.  It
does not generate environments, run PPO, or mutate training code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REPO = Path("/home/chen/RLPFN")
ART = REPO / "artifacts"
DEFAULT_BASE_DIR = ART / "phase2_survival_source_label_multiseed_u4_s128_0519"
DEFAULT_OUTPUT_DIR = ART / "phase2_survival_full_q90_u4_multiseed_opt_parity_0519"
DEFAULT_PACK_FIT_REPORT = (
    REPO
    / "reinforce-terminal-explore"
    / "artifacts"
    / "phase2_m4_live_survival_source_label_pack_fit_parity_n2_s32_0519"
    / "report.json"
)
DEFAULT_SURVIVAL_AUDIT = (
    REPO
    / "reinforce-terminal-explore"
    / "artifacts"
    / "phase2_survival_component_reconstruction_audit_0519"
    / "survival_component_reconstruction_audit.json"
)

EXPECTED_RUNS = {
    "full": (9821, 9921, 10021),
    "q90": (9822, 9922, 10022),
}
EXPECTED_UPDATES = 4
TARGET_KL = 0.03
MAX_CLIP_FRACTION = 0.05


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _f(value: Any, default: float = float("nan")) -> float:
    if _finite(value):
        return float(value)
    return default


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _count_true(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if bool(value))


def _count_positive(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    return sum(1 for value in values if _finite(value) and float(value) > 0.0)


def _summarize_run(run_dir: Path, *, scope: str, seed: int) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "prior_progress.jsonl"
    summary = _read_json(summary_path)
    rows = _read_jsonl(progress_path)
    update_idxs = [row.get("update_idx") for row in rows]
    exceptions = [_get(row, "update.exception") for row in rows]

    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    stats_first = _get(first, "pack_summary.stats", {}) or {}
    stats_last = _get(last, "pack_summary.stats", {}) or {}
    logger_values = [_get(row, "update.logger", {}) or {} for row in rows]
    param_values = [_get(row, "update.param_delta", {}) or {} for row in rows]
    reward_components = _get(last, "reward_component_summary", {}) or {}
    collector = _get(last, "collector", {}) or {}

    kl_values = [_f(logger.get("train/approx_kl")) for logger in logger_values]
    clip_values = [_f(logger.get("train/clip_fraction")) for logger in logger_values]
    ev_values = [_f(logger.get("train/explained_variance")) for logger in logger_values]
    param_l2_values = [_f(param.get("param_delta_l2")) for param in param_values]
    update_wall_s = [_f(logger.get("train/update_wall_time_sec")) for logger in logger_values]

    survival_enabled = collector.get("survival_reward_enabled")
    survival_weight = collector.get("survival_reward_weight")
    survival_on_count = _count_true(survival_enabled)
    survival_weight_positive_count = _count_positive(survival_weight)
    n_envs = int(_get(last, "collector.n_envs", 0) or 0)

    raw_delta = _f(stats_last.get("raw_reward_return_mean")) - _f(
        stats_first.get("raw_reward_return_mean")
    )
    train_delta = _f(stats_last.get("reward_return_mean")) - _f(
        stats_first.get("reward_return_mean")
    )
    env_component_return = _f(reward_components.get("reward_env_return_mean"))
    survival_component_return = _f(reward_components.get("reward_survival_return_mean"), 0.0)
    component_total_return = _f(reward_components.get("reward_return_mean"))
    survival_abs_share = abs(survival_component_return) / max(
        1e-8,
        abs(env_component_return)
        + abs(_f(reward_components.get("reward_ctrl_return_mean"), 0.0))
        + abs(_f(reward_components.get("reward_terminal_bonus_return_mean"), 0.0))
        + abs(survival_component_return),
    )

    arm = ((summary.get("arms") or [{}])[0]) if isinstance(summary.get("arms"), list) else {}
    progress_clean_ok = (
        len(rows) == EXPECTED_UPDATES
        and update_idxs == list(range(EXPECTED_UPDATES))
        and all(exc is None for exc in exceptions)
    )
    ppo_health_ok = (
        all(_finite(x) for x in kl_values + clip_values + ev_values)
        and max(abs(x) for x in kl_values) <= TARGET_KL
        and max(clip_values) <= MAX_CLIP_FRACTION
    )
    param_delta_ok = all(_finite(x) and x > 0.0 for x in param_l2_values)
    reward_health_ok = (
        _finite(stats_last.get("reward_std"))
        and _finite(stats_last.get("raw_reward_std"))
        and float(stats_last.get("reward_std")) > 0.0
        and float(stats_last.get("raw_reward_std")) > 0.0
        and _finite(env_component_return)
        and _finite(component_total_return)
    )
    survival_axis_ok = (
        n_envs == 64
        and survival_on_count == 32
        and survival_weight_positive_count == 32
    )
    hard_pass_ok = bool(summary.get("hard_pass")) and bool(arm.get("hard_pass"))
    fixed_digest_ok = bool(arm.get("env_group_fixed_pass")) and int(
        arm.get("env_group_digest_unique_count", -1)
    ) == 1

    return {
        "scope": scope,
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_hard_pass": bool(summary.get("hard_pass")),
        "arm_hard_pass": bool(arm.get("hard_pass")),
        "updates": len(rows),
        "update_idxs": update_idxs,
        "progress_clean_ok": progress_clean_ok,
        "fixed_digest_ok": fixed_digest_ok,
        "env_group_digest_first": arm.get("env_group_digest_first"),
        "kl_max": max(kl_values) if kl_values else None,
        "clip_max": max(clip_values) if clip_values else None,
        "ev_mean": mean(ev_values) if ev_values else None,
        "ev_last": ev_values[-1] if ev_values else None,
        "param_delta_l2_last": param_l2_values[-1] if param_l2_values else None,
        "param_delta_l2_mean": mean(param_l2_values) if param_l2_values else None,
        "update_wall_time_sec_mean": mean(update_wall_s) if update_wall_s else None,
        "raw_reward_return_mean_first": _f(stats_first.get("raw_reward_return_mean")),
        "raw_reward_return_mean_last": _f(stats_last.get("raw_reward_return_mean")),
        "raw_reward_return_mean_delta": raw_delta,
        "training_reward_return_mean_first": _f(stats_first.get("reward_return_mean")),
        "training_reward_return_mean_last": _f(stats_last.get("reward_return_mean")),
        "training_reward_return_mean_delta": train_delta,
        "raw_reward_std_last": _f(stats_last.get("raw_reward_std")),
        "training_reward_std_last": _f(stats_last.get("reward_std")),
        "reward_env_return_mean_last": env_component_return,
        "reward_survival_return_mean_last": survival_component_return,
        "reward_component_return_mean_last": component_total_return,
        "survival_abs_component_share_last": survival_abs_share,
        "survival_on_count": survival_on_count,
        "survival_off_count": n_envs - survival_on_count if n_envs else None,
        "survival_weight_positive_count": survival_weight_positive_count,
        "hard_pass_ok": hard_pass_ok,
        "ppo_health_ok": ppo_health_ok,
        "param_delta_ok": param_delta_ok,
        "reward_health_ok": reward_health_ok,
        "survival_axis_ok": survival_axis_ok,
        "run_gate_pass": all(
            [
                progress_clean_ok,
                hard_pass_ok,
                fixed_digest_ok,
                ppo_health_ok,
                param_delta_ok,
                reward_health_ok,
                survival_axis_ok,
            ]
        ),
    }


def _group_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(run[key]) for run in runs if _finite(run.get(key))]

    return {
        "count": len(runs),
        "all_run_gate_pass": all(bool(run.get("run_gate_pass")) for run in runs),
        "raw_reward_return_mean_delta_avg": mean(vals("raw_reward_return_mean_delta")),
        "raw_reward_return_mean_delta_min": min(vals("raw_reward_return_mean_delta")),
        "training_reward_return_mean_delta_avg": mean(vals("training_reward_return_mean_delta")),
        "training_reward_return_mean_delta_min": min(vals("training_reward_return_mean_delta")),
        "kl_max": max(vals("kl_max")),
        "clip_max": max(vals("clip_max")),
        "ev_mean_avg": mean(vals("ev_mean")),
        "param_delta_l2_last_avg": mean(vals("param_delta_l2_last")),
        "training_reward_std_last_avg": mean(vals("training_reward_std_last")),
        "raw_reward_std_last_avg": mean(vals("raw_reward_std_last")),
        "survival_abs_component_share_last_avg": mean(vals("survival_abs_component_share_last")),
        "survival_abs_component_share_last_max": max(vals("survival_abs_component_share_last")),
    }


def _pack_fit_terms(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    grad = _get(report, "prior_pack_ppo_update_parity.gradient_diff", {}) or {}
    loss = _get(report, "prior_pack_ppo_update_parity.loss_diff", {}) or {}
    return {
        "path": str(path),
        "hard_pass": bool(report.get("hard_pass")),
        "grad_delta_l2": grad.get("grad_delta_l2"),
        "grad_delta_max_abs": grad.get("grad_delta_max_abs"),
        "policy_loss_delta": _get(loss, "policy_loss.delta"),
        "value_loss_delta": _get(loss, "value_loss.delta"),
        "total_loss_delta": _get(loss, "total_loss.delta"),
        "approx_kl_delta": _get(loss, "approx_kl.delta"),
        "clip_fraction_delta": _get(loss, "clip_fraction.delta"),
    }


def _survival_audit_terms(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    read = report.get("read", {}) or {}
    return {
        "path": str(path),
        "hard_pass": bool(read.get("hard_pass")),
        "component_logging_pass": bool(read.get("component_logging_pass")),
        "reconstruction_pass": bool(read.get("reconstruction_pass")),
        "base_preservation_pass": bool(read.get("base_preservation_pass")),
        "source_label_sampler_authorized": bool(
            read.get("source_label_sampler_authorized_after_this_audit")
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M4 Survival Full/Q90 Multi-seed U4 Opt Parity",
        "",
        f"decision: `{report['decision']['survival_full_q90_multiseed_u4_opt_parity_pass']}`",
        "",
        "| scope | seeds | raw delta avg | train delta avg | KL max | clip max | EV avg | param delta avg | survival share avg/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, summary in report["groups"].items():
        seeds = ",".join(str(run["seed"]) for run in report["runs"] if run["scope"] == scope)
        lines.append(
            "| {scope} | {seeds} | {raw:.6g} | {train:.6g} | {kl:.3g} | {clip:.3g} | {ev:.6g} | {param:.6g} | {share_avg:.3g}/{share_max:.3g} |".format(
                scope=scope,
                seeds=seeds,
                raw=summary["raw_reward_return_mean_delta_avg"],
                train=summary["training_reward_return_mean_delta_avg"],
                kl=summary["kl_max"],
                clip=summary["clip_max"],
                ev=summary["ev_mean_avg"],
                param=summary["param_delta_l2_last_avg"],
                share_avg=summary["survival_abs_component_share_last_avg"],
                share_max=summary["survival_abs_component_share_last_max"],
            )
        )
    lines.extend(
        [
            "",
            "## Gate Terms",
            "",
            "```json",
            json.dumps(report["gate_terms"], indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for scope, seeds in EXPECTED_RUNS.items():
        for seed in seeds:
            run_dir = base_dir / f"seed_{seed}_{scope}"
            runs.append(_summarize_run(run_dir, scope=scope, seed=seed))

    groups = {
        scope: _group_summary([run for run in runs if run["scope"] == scope])
        for scope in EXPECTED_RUNS
    }
    pack_fit = _pack_fit_terms(Path(args.pack_fit_report).expanduser().resolve())
    survival_audit = _survival_audit_terms(Path(args.survival_audit).expanduser().resolve())
    gate_terms = {
        "all_expected_runs_present_and_clean": all(run["progress_clean_ok"] for run in runs),
        "all_runner_hard_pass": all(run["hard_pass_ok"] for run in runs),
        "all_fixed_digest_ok": all(run["fixed_digest_ok"] for run in runs),
        "all_ppo_health_ok": all(run["ppo_health_ok"] for run in runs),
        "all_param_delta_ok": all(run["param_delta_ok"] for run in runs),
        "all_reward_health_ok": all(run["reward_health_ok"] for run in runs),
        "all_survival_axis_ok": all(run["survival_axis_ok"] for run in runs),
        "pack_fit_hard_pass": bool(pack_fit["hard_pass"]),
        "survival_audit_hard_pass": bool(survival_audit["hard_pass"]),
        "survival_base_preservation_pass": bool(survival_audit["base_preservation_pass"]),
        "post_delta_observed_not_gate": True,
    }
    decision = {
        "survival_full_q90_multiseed_u4_opt_parity_pass": all(gate_terms.values()),
        "eligible_for_next_stage": all(gate_terms.values()),
        "longtrain_decision": "not_started_by_this_report",
    }
    report = {
        "schema": "phase2_survival_full_q90_multiseed_u4_opt_parity.v1",
        "base_dir": str(base_dir),
        "expected_runs": EXPECTED_RUNS,
        "runs": runs,
        "groups": groups,
        "pack_fit": pack_fit,
        "survival_audit": survival_audit,
        "gate_terms": gate_terms,
        "decision": decision,
    }
    report = _json_safe(report)
    json_path = out_dir / "survival_full_q90_u4_multiseed_opt_parity_report.json"
    md_path = out_dir / "survival_full_q90_u4_multiseed_opt_parity_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pack-fit-report", default=str(DEFAULT_PACK_FIT_REPORT))
    parser.add_argument("--survival-audit", default=str(DEFAULT_SURVIVAL_AUDIT))
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "hard_pass": report["decision"]["survival_full_q90_multiseed_u4_opt_parity_pass"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "gate_terms": report["gate_terms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
