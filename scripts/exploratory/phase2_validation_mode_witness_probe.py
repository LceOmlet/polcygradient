#!/usr/bin/env python
"""Validation-mode witness probe for targeted prior mismatch hypotheses.

This is an exploratory, read-only probe.  It does not mutate exact SCM defaults,
does not install generator wrappers, and does not run PPO.  Its purpose is to
separate environment-side validation signatures from PPO-side sufficiency tests:

* InvertedPendulum/InvertedDoublePendulum are checked for a terminal/survival
  necessary signature: return tracks episode length/survival reward.
* Ant is checked as a locomotion/energy signature: episodes survive to horizon
  while control cost/forward reward dominate the residual return.

Passing a necessary signature is not treated as sufficient.  The script reports
the next counterfactual witness needed before any milestone change.
"""

from __future__ import annotations

import argparse
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


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": float(sum(finite) / len(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def parse_return_log(path: str | Path) -> dict[str, dict[int, dict[str, float]]]:
    """Parse fit_model-style validation return lines from a text log."""

    env_epochs: dict[str, dict[int, dict[str, float]]] = {}
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RETURN_RE.match(raw_line.strip())
        if not match:
            continue
        epoch = int(match.group("epoch"))
        env = str(match.group("env"))
        metric = str(match.group("metric"))
        value = _as_float(match.group("value"))
        if value is None:
            continue
        env_epochs.setdefault(env, {}).setdefault(epoch, {})[metric] = float(value)
    return env_epochs


def _metric_series(env_epochs: dict[int, dict[str, float]], metric: str) -> list[float]:
    return [
        float(metrics[metric])
        for _, metrics in sorted(env_epochs.items())
        if metric in metrics and math.isfinite(float(metrics[metric]))
    ]


def _inverted_witness(env_epochs: dict[int, dict[str, float]]) -> dict[str, Any]:
    returns = _metric_series(env_epochs, "return_mean")
    lengths = _metric_series(env_epochs, "len_mean")
    survives = _metric_series(env_epochs, "reward_survive_mean")
    return_len_corr = _pearson(returns, lengths)
    return_survive_corr = _pearson(returns, survives)
    len_stats = _summary(lengths)
    survive_stats = _summary(survives)
    necessary_pass = bool(
        len(returns) >= 3
        and (
            (return_len_corr is not None and return_len_corr >= 0.80)
            or (return_survive_corr is not None and return_survive_corr >= 0.80)
        )
        and (len_stats.get("max") is None or float(len_stats["max"]) < 200.0)
    )
    return {
        "target_mechanism": "terminal_survival_return",
        "necessary_signature_pass": necessary_pass,
        "return_len_corr": return_len_corr,
        "return_survive_corr": return_survive_corr,
        "len_mean": len_stats,
        "reward_survive_mean": survive_stats,
        "sufficiency_status": "not_proven_requires_action_counterfactual_and_ppo_advantage_witness",
        "required_counterfactual": [
            "same initial state/noise with zero/random/policy actions changes first_done_step",
            "official PPO returns/advantages increase with future survival length",
            "short PPO smoke improves raw eval length/return on fixed terminal-sensitive list",
        ],
    }


def _ant_witness(env_epochs: dict[int, dict[str, float]], *, horizon: float) -> dict[str, Any]:
    returns = _metric_series(env_epochs, "return_mean")
    lengths = _metric_series(env_epochs, "len_mean")
    survives = _metric_series(env_epochs, "reward_survive_mean")
    ctrls = _metric_series(env_epochs, "reward_ctrl_mean")
    forwards = _metric_series(env_epochs, "reward_forward_mean")
    len_stats = _summary(lengths)
    ctrl_stats = _summary(ctrls)
    forward_stats = _summary(forwards)
    survived_horizon = bool(
        len_stats.get("min") is not None and float(len_stats["min"]) >= 0.95 * float(horizon)
    )
    ctrl_return_corr = _pearson(returns, ctrls)
    forward_return_corr = _pearson(returns, forwards)
    return {
        "target_mechanism": "locomotion_energy_tradeoff",
        "terminal_not_primary_signature_pass": survived_horizon,
        "len_mean": len_stats,
        "reward_survive_mean": _summary(survives),
        "reward_ctrl_mean": ctrl_stats,
        "reward_forward_mean": forward_stats,
        "return_ctrl_corr": ctrl_return_corr,
        "return_forward_corr": forward_return_corr,
        "sufficiency_status": "not_proven_requires_forward_ctrl_counterfactual_and_short_ppo_witness",
        "required_counterfactual": [
            "action-norm changes predict control penalty without collapsing survival",
            "state/action perturbations expose learnable forward reward gradients",
            "short PPO smoke improves forward/control tradeoff while Ant length stays near horizon",
        ],
    }


def _load_done_audit(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data


def build_witness_report(
    *,
    validation_log: str | Path,
    done_audit_json: str | None = None,
    ant_horizon: float = 1000.0,
) -> dict[str, Any]:
    env_epochs = parse_return_log(validation_log)
    inverted_envs = ["InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
    inverted = {
        env_id: _inverted_witness(env_epochs.get(env_id, {}))
        for env_id in inverted_envs
        if env_id in env_epochs
    }
    ant = _ant_witness(env_epochs.get("Ant-v5", {}), horizon=float(ant_horizon)) if "Ant-v5" in env_epochs else {}
    done_audit = _load_done_audit(done_audit_json)

    inverted_pass_count = sum(
        1 for item in inverted.values() if bool(item.get("necessary_signature_pass"))
    )
    ant_terminal_not_primary = bool(ant.get("terminal_not_primary_signature_pass", False))
    selected = "terminal_survival_action_sensitive_counterfactual"
    rationale = (
        "Inverted validation returns expose the clearest necessary terminal/survival "
        "signature, while Ant survives to horizon and should be handled as a separate "
        "locomotion/energy tradeoff."
    )
    if inverted_pass_count == 0 and ant:
        selected = "locomotion_energy_counterfactual"
        rationale = "No Inverted terminal/survival necessary signature passed; Ant is the remaining explicit witness target."

    return {
        "probe": "phase2_validation_mode_witness_probe",
        "exploratory_only": True,
        "mutates_exact_scm": False,
        "mutates_ppo": False,
        "validation_log": str(Path(validation_log).expanduser().resolve()),
        "done_audit_json": None if done_audit_json is None else str(Path(done_audit_json).expanduser().resolve()),
        "contract": {
            "necessary_signatures_do_not_imply_milestone_change": True,
            "environment_side_consumes_actions_only": True,
            "ppo_side_must_be_checked_with_official_buffer_returns_advantages": True,
            "formal_gated_reward_path_balance_remains_reward_path_only": True,
        },
        "selected_next_witness": selected,
        "selection_rationale": rationale,
        "inverted_terminal_survival": inverted,
        "ant_locomotion_energy": ant,
        "done_audit_context": done_audit,
        "sufficient_to_change_milestone": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-log", required=True)
    parser.add_argument("--done-audit-json", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ant-horizon", type=float, default=1000.0)
    args = parser.parse_args(argv)

    report = build_witness_report(
        validation_log=args.validation_log,
        done_audit_json=args.done_audit_json,
        ant_horizon=float(args.ant_horizon),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(output),
        "selected_next_witness": report["selected_next_witness"],
        "sufficient_to_change_milestone": report["sufficient_to_change_milestone"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
