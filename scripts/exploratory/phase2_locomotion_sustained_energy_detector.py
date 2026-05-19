#!/usr/bin/env python
"""Read-only locomotion/sustained-energy mode coverage detector.

This detector is intentionally conservative.  It does not mutate the exact SCM
prior, fit_model, PPO, checkpoints, or Gym environments.  Its purpose is to make
one missing row in the mode coverage table explicit before any generator repair:

* Gym side: identify validation envs where episodes already survive and return
  is dominated by sustained forward/control tradeoff rather than terminal length.
* Prior side: use the existing prior action-mode coverage report as a proxy for
  homologous state-conditioned env-reward improvement and flag what is still
  missing for a full detector.

The prior proxy is not accepted as a milestone criterion.  It is a triage table
that tells us whether the current prior has enough candidates to justify a more
expensive same-rollout action-temporal detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


RETURN_METRICS = (
    "context_len_before_eval_mean",
    "explore_rollout_count_mean",
    "explore_rollout_len_mean",
    "len_mean",
    "make_failed",
    "return_mean",
    "reward_contact_mean",
    "reward_ctrl_mean",
    "reward_dist_mean",
    "reward_forward_mean",
    "reward_survive_mean",
)

RETURN_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+return_(?P<env>.+)_(?P<metric>"
    + "|".join(re.escape(metric) for metric in RETURN_METRICS)
    + r")\s+(?P<value>[-+0-9.eE]+)\s*$"
)

DEFAULT_VALIDATION_LOG = (
    "/home/chen/RLPFN/health/log/"
    "rlpfn_ppopackpriormilestonegated_reward_path_balance_05_07_2026_01_53_23.log"
)
DEFAULT_PRIOR_ACTION_COVERAGE = (
    "/home/chen/RLPFN/artifacts/phase2_prior_action_mode_coverage_probe_0508/"
    "gated_full_q90_n64_s256_v2.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/phase2_locomotion_sustained_energy_detector_0510"
)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _summary(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    finite_sorted = sorted(finite)

    def q(p: float) -> float:
        if len(finite_sorted) == 1:
            return finite_sorted[0]
        idx = p * (len(finite_sorted) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        w = idx - lo
        return float((1.0 - w) * finite_sorted[lo] + w * finite_sorted[hi])

    return {
        "count": len(finite),
        "mean": float(sum(finite) / len(finite)),
        "min": float(min(finite)),
        "q10": q(0.10),
        "q50": q(0.50),
        "q90": q(0.90),
        "max": float(max(finite)),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    paired = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(paired) < 3:
        return None
    x_vals = [x for x, _ in paired]
    y_vals = [y for _, y in paired]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in paired)
    return float(cov / math.sqrt(x_var * y_var))


def parse_validation_log(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    env_epochs: dict[str, dict[int, dict[str, float]]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RETURN_RE.match(line.strip())
        if not match:
            continue
        value = _finite_float(match.group("value"))
        if value is None:
            continue
        env = str(match.group("env"))
        epoch = int(match.group("epoch"))
        metric = str(match.group("metric"))
        env_epochs.setdefault(env, {}).setdefault(epoch, {})[metric] = float(value)
    return env_epochs


def _series(env_epochs: dict[int, dict[str, float]], metric: str) -> list[float]:
    return [
        float(row[metric])
        for _, row in sorted(env_epochs.items())
        if metric in row and math.isfinite(float(row[metric]))
    ]


def gym_locomotion_rows(
    env_epochs: dict[str, dict[int, dict[str, float]]],
    *,
    horizon: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env, epochs in sorted(env_epochs.items()):
        has_forward = any("reward_forward_mean" in row for row in epochs.values())
        has_ctrl = any("reward_ctrl_mean" in row for row in epochs.values())
        if not has_forward:
            continue
        returns = _series(epochs, "return_mean")
        lengths = _series(epochs, "len_mean")
        forwards = _series(epochs, "reward_forward_mean")
        ctrls = _series(epochs, "reward_ctrl_mean")
        survives = _series(epochs, "reward_survive_mean")
        len_stats = _summary(lengths)
        forward_stats = _summary(forwards)
        ctrl_stats = _summary(ctrls)
        survived_horizon = bool(
            len_stats.get("min") is not None and float(len_stats["min"]) >= 0.95 * float(horizon)
        )
        terminal_not_primary = bool(survived_horizon or not survives)
        forward_corr = _pearson(returns, forwards)
        ctrl_corr = _pearson(returns, ctrls)
        forward_nontrivial = bool(
            forward_stats.get("count", 0)
            and max(abs(float(forward_stats["min"])), abs(float(forward_stats["max"]))) >= 1e-3
        )
        ctrl_nontrivial = bool(
            ctrl_stats.get("count", 0)
            and max(abs(float(ctrl_stats["min"])), abs(float(ctrl_stats["max"]))) >= 1e-5
        )
        sustained_signature = bool(
            terminal_not_primary
            and (forward_nontrivial or ctrl_nontrivial)
            and (
                forward_corr is None
                or abs(float(forward_corr)) >= 0.25
                or ctrl_corr is None
                or abs(float(ctrl_corr)) >= 0.25
            )
        )
        rows.append(
            {
                "env": env,
                "gym_signature": "sustained_locomotion_energy_tradeoff",
                "sustained_signature": sustained_signature,
                "terminal_not_primary": terminal_not_primary,
                "survived_horizon": survived_horizon,
                "len_mean": len_stats,
                "return_mean": _summary(returns),
                "reward_forward_mean": forward_stats,
                "reward_ctrl_mean": ctrl_stats,
                "reward_survive_mean": _summary(survives),
                "return_forward_corr": forward_corr,
                "return_ctrl_corr": ctrl_corr,
                "read": (
                    "strong gym witness"
                    if sustained_signature
                    else "weak/partial gym witness; needs action counterfactual"
                ),
            }
        )
    return rows


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prior_proxy_rows(action_coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in action_coverage.get("lists", []):
        label = str(item.get("label", "unknown"))
        classification = item.get("classification", {})
        coverage = classification.get("coverage", {})
        per_env = classification.get("per_env", [])
        n = int(coverage.get("n_envs", len(per_env)) or 0)
        state_env_count = int(coverage.get("state_conditioned_env_reward_witness_count", 0) or 0)
        state_total_count = int(coverage.get("state_conditioned_energy_injection_witness_count", 0) or 0)
        low_energy_best = int(coverage.get("low_energy_global_best_count", 0) or 0)
        state_best = int(coverage.get("state_conditioned_global_best_count", 0) or 0)
        sign_count = int(coverage.get("sign_or_phase_sensitive_present_count", 0) or 0)
        # Conservative proxy: state-conditioned env reward exists and the list is
        # not purely static/low-energy dominated.  This is not a proof of
        # locomotion, only a signal that a full temporal detector is worth running.
        proxy_count = 0
        for row in per_env:
            state_env = bool(row.get("state_conditioned_env_reward_witness_present", False))
            low_energy_global = bool(str(row.get("best_mode", "")) in {"zero", "half_random"})
            state_global = bool(str(row.get("best_mode", "")).startswith("obs_linear"))
            sign = bool(row.get("sign_or_phase_sensitive_present", False))
            if state_env and (state_global or sign) and not low_energy_global:
                proxy_count += 1
        rows.append(
            {
                "scope": label,
                "prior_signature_proxy": "sustained_state_conditioned_env_reward_tradeoff",
                "n": n,
                "state_conditioned_env_reward_count": state_env_count,
                "state_conditioned_env_reward_rate": None if n <= 0 else state_env_count / n,
                "state_conditioned_total_count": state_total_count,
                "state_conditioned_total_rate": None if n <= 0 else state_total_count / n,
                "state_conditioned_global_best_count": state_best,
                "state_conditioned_global_best_rate": None if n <= 0 else state_best / n,
                "low_energy_global_best_count": low_energy_best,
                "low_energy_global_best_rate": None if n <= 0 else low_energy_best / n,
                "sign_phase_count": sign_count,
                "sign_phase_rate": None if n <= 0 else sign_count / n,
                "sustained_proxy_count": proxy_count,
                "sustained_proxy_rate": None if n <= 0 else proxy_count / n,
                "read": (
                    "proxy present; run full temporal detector"
                    if n > 0 and proxy_count / n >= 0.15
                    else "proxy weak; inspect whether prior lacks sustained mode or proxy is too strict"
                ),
                "missing_for_full_detector": [
                    "same-env temporal action persistence",
                    "first_done / terminal invariance under action families",
                    "progress-like state displacement or env-reward basin improvement",
                    "fixed-list PPO trainability stratified by proxy",
                ],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                flat[key] = value
            writer.writerow(flat)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    validation_log = Path(args.validation_log).expanduser()
    prior_action_coverage = Path(args.prior_action_coverage).expanduser()
    env_epochs = parse_validation_log(validation_log)
    gym_rows = gym_locomotion_rows(env_epochs, horizon=float(args.horizon))
    prior_rows = prior_proxy_rows(_load_json(prior_action_coverage))
    return {
        "analysis_entry": "phase2_locomotion_sustained_energy_detector",
        "contract": {
            "reads_existing_artifacts_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_run_training": True,
            "exploratory_only": True,
        },
        "definition": {
            "gym_side": (
                "episodes are terminal-insensitive or horizon-stable, while return has nontrivial "
                "forward/control components; this is a necessary witness, not a sufficient controller proof"
            ),
            "prior_proxy": (
                "state-conditioned policy family improves env reward and is not simply a low-energy global best; "
                "this is a proxy because existing artifact lacks temporal action persistence and first_done counterfactuals"
            ),
        },
        "inputs": {
            "validation_log": str(validation_log.resolve()),
            "prior_action_coverage": str(prior_action_coverage.resolve()),
        },
        "gym_rows": gym_rows,
        "prior_proxy_rows": prior_rows,
        "decision": {
            "repair_generator_now": False,
            "reason": (
                "This detector is a coverage/readout audit.  It should be used to decide whether a full "
                "temporal prior detector is needed before any generator-level change."
            ),
            "next_min_detector": (
                "Run a same-rollout prior temporal detector with action persistence, terminal invariance, "
                "and env-reward/state-displacement improvement; then stratify fixed-list PPO trainability."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-log", default=DEFAULT_VALIDATION_LOG)
    parser.add_argument("--prior-action-coverage", default=DEFAULT_PRIOR_ACTION_COVERAGE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=float, default=1000.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    report_path = out_dir / "locomotion_sustained_energy_detector_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "gym_locomotion_rows.csv", report["gym_rows"])
    write_csv(out_dir / "prior_locomotion_proxy_rows.csv", report["prior_proxy_rows"])

    md = [
        "# Locomotion / Sustained-Energy Detector",
        "",
        "Read-only detector. It does not modify exact SCM, fit_model, PPO, or checkpoints.",
        "",
        "## Decision",
        "",
        f"- repair_generator_now: `{report['decision']['repair_generator_now']}`",
        f"- reason: {report['decision']['reason']}",
        f"- next_min_detector: {report['decision']['next_min_detector']}",
        "",
        "## Gym Rows",
        "",
        "| env | sustained_signature | terminal_not_primary | return_forward_corr | return_ctrl_corr | read |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["gym_rows"]:
        md.append(
            "| {env} | {sustained_signature} | {terminal_not_primary} | {return_forward_corr} | {return_ctrl_corr} | {read} |".format(
                **row
            )
        )
    md.extend(
        [
            "",
            "## Prior Proxy Rows",
            "",
            "| scope | sustained_proxy | state_env_reward | state_global_best | low_energy_global_best | sign_phase | read |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["prior_proxy_rows"]:
        md.append(
            "| {scope} | {sustained_proxy_count}/{n} ({sustained_proxy_rate:.3f}) | "
            "{state_conditioned_env_reward_count}/{n} ({state_conditioned_env_reward_rate:.3f}) | "
            "{state_conditioned_global_best_count}/{n} ({state_conditioned_global_best_rate:.3f}) | "
            "{low_energy_global_best_count}/{n} ({low_energy_global_best_rate:.3f}) | "
            "{sign_phase_count}/{n} ({sign_phase_rate:.3f}) | {read} |".format(**row)
        )
    (out_dir / "locomotion_sustained_energy_detector.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
