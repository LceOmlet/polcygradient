#!/usr/bin/env python
"""Long-run preflight gate for profile prior-pack candidates.

This is a read-only aggregator.  It intentionally does not sample
environments, train PPO, modify selectors, or patch the prior.  The point is
to keep long fit_model submission behind one explicit gate instead of chasing
one metric at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTOR_REPORT = (
    ART
    / "phase2_profile_band_selector_v2n_prior_pack_uniform_terminal_gym_baseline_modes_seed9730_0515"
    / "profile_band_selector_report.json"
)
DEFAULT_SELECTOR_CSV = DEFAULT_SELECTOR_REPORT.parent / "selected_profile_band_prior_envs.csv"
DEFAULT_SPECTRAL_JSON = (
    ART / "phase2_profile_v2n_spectral_rank_audit_0515/profile_spectral_rank_audit.json"
)
DEFAULT_RELATIVE_JSON = (
    ART / "phase2_profile_v2n_relative_complexity_audit_0515/profile_relative_complexity_audit.json"
)
DEFAULT_PACK_FIT_PARITY_JSON = (
    ART
    / "phase2_profile_v2n_uniform_terminal_gym_baseline_pack_fit_parity_n2_s32_0515"
    / "profile_v2e_pack_fit_model_parity_report.json"
)
DEFAULT_ENV_PROPERTY_PARITY_JSON = (
    ART / "phase2_profile_v2n_property_parity_n16_s64_0515/m3_env_property_parity_report.json"
)
DEFAULT_PPO_CORE_JSON = (
    ART / "phase2_m3_recheck_official_recurrent_ppo_core_parity_0513/parity_report.json"
)
DEFAULT_PPO_REAL_ROLLOUT_JSON = (
    ART
    / "phase2_m3_recheck_official_recurrent_ppo_real_rollout_update_parity_0513"
    / "real_rollout_update_parity_report.json"
)
DEFAULT_PRIOR_MILESTONE_JSON = (
    ART
    / "phase2_prior_milestone_all_parity_gate_terminal80_warmup_0510"
    / "prior_milestone_all_parity_gate_report.json"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2n_longrun_preflight_gate_0515"

RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")
MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _pass(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "details": details or {}}


def _terminal_bins_from_csv(rows: list[dict[str, str]], rule: str) -> dict[str, Any]:
    values = [
        _safe_float(row.get("terminal_reset_count_target"))
        for row in rows
        if str(row.get("rule")) == rule
    ]
    bins = []
    edges = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0)]
    for idx, (lo, hi) in enumerate(edges):
        count = sum(
            (lo <= value <= hi) if idx == 0 else (lo < value <= hi)
            for value in values
        )
        bins.append(int(count))
    expected = len(values) / len(edges) if edges else 0.0
    max_abs_from_uniform = max([abs(count - expected) for count in bins] or [0.0])
    return {
        "n": int(len(values)),
        "bins_0_20_40_60_80": bins,
        "mean": float(sum(values) / len(values)) if values else None,
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "expected_per_bin": expected,
        "max_abs_count_from_uniform": float(max_abs_from_uniform),
        "within_0_80": bool(values and min(values) >= 0.0 and max(values) <= 80.0),
    }


def _selector_gate(selector: dict[str, Any], selector_rows: list[dict[str, str]]) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    for rule in RULES:
        payload = selector.get("rules", {}).get(rule, {})
        summary = payload.get("selected_summary", {}) or {}
        band_details = payload.get("band_details", {}) or {}
        opt = band_details.get("prior_pack_optimizability_guard", {}) or {}
        terminal = _terminal_bins_from_csv(selector_rows, rule)
        mode_counts = {
            key: int(summary.get(f"{key}_count", 0))
            for key in MODE_KEYS
        }
        mode_pass = all(
            bool((band_details.get(key, {}) or {}).get("passed", False))
            for key in MODE_KEYS
        )
        terminal_distribution_pass = bool(
            ((opt.get("terminal_distribution_guard", {}) or {}).get("passed", False))
        )
        dominant_rate = _safe_float(summary.get("joint_profile_dominant_rate"))
        active_cells = int(summary.get("joint_profile_active_cells", 0))
        strict = band_details.get("strict_feedback_subprofile_branch", {}) or {}
        strict_pass = bool(strict.get("passed", True))
        opt_pass = bool(opt.get("passed", False))
        # The selector's hard min/max bins are the source of truth; this
        # additional uniformity bound catches accidental skew if a future report
        # is missing terminal_distribution_guard details.
        empirical_uniform_pass = bool(
            terminal["within_0_80"]
            and terminal["n"] == 64
            and terminal["max_abs_count_from_uniform"] <= 4.0
        )
        passed = bool(
            payload.get("guard_pass", False)
            and mode_pass
            and terminal_distribution_pass
            and empirical_uniform_pass
            and strict_pass
            and opt_pass
            and dominant_rate <= 0.25 + 1e-12
            and active_cells >= 12
        )
        rules[rule] = {
            "pass": passed,
            "selector_guard_pass": bool(payload.get("guard_pass", False)),
            "mode_pass": mode_pass,
            "mode_counts": mode_counts,
            "strict_feedback_count": int(summary.get("strict_feedback_count", 0)),
            "strict_pass": strict_pass,
            "pendulum_settle_effective_support_count": int(
                summary.get("pendulum_settle_effective_support_count", 0)
            ),
            "optimizability_proxy_pass": opt_pass,
            "optimizability_proxy_count": int(opt.get("selected_proxy_count", 0)),
            "optimizability_proxy_min": int(opt.get("min_proxy_count", 0)),
            "terminal_distribution_pass": terminal_distribution_pass,
            "terminal_uniformity_empirical_pass": empirical_uniform_pass,
            "terminal": terminal,
            "joint_profile_dominant_rate": dominant_rate,
            "joint_profile_active_cells": active_cells,
        }
    return {
        "pass": bool(rules and all(item["pass"] for item in rules.values())),
        "rules": rules,
    }


def _optimizability_gate(spectral: dict[str, Any]) -> dict[str, Any]:
    gym = spectral["runs"]["gym"]
    prior_full = spectral["runs"]["prior_full"]
    prior_q90 = spectral["runs"]["prior_q90"]
    gym_delta = _safe_float(gym.get("delta_mean"))
    gym_improved = int(gym.get("improved_count", 0))
    runs: dict[str, Any] = {}
    for name, item in (("prior_full", prior_full), ("prior_q90", prior_q90)):
        delta = _safe_float(item.get("delta_mean"))
        improved = int(item.get("improved_count", 0))
        comp = spectral.get("comparisons", {}).get(name, {}) or {}
        delta_ratio = delta / gym_delta if abs(gym_delta) > 1e-12 else None
        passed = bool(
            delta > 0.0
            and (delta_ratio is None or delta_ratio >= 0.90)
            and improved >= gym_improved - 1
            and bool(comp.get("common_behavior_rank_not_below_gym_90pct", False))
        )
        runs[name] = {
            "pass": passed,
            "delta_mean": delta,
            "delta_ratio_vs_gym": delta_ratio,
            "improved_count": improved,
            "gym_improved_count": gym_improved,
            "improved_count_margin_vs_gym": improved - gym_improved,
            "improved_effective_rank_vs_gym": comp.get("improved_effective_rank_vs_gym"),
        }
    return {
        "pass": bool(all(item["pass"] for item in runs.values())),
        "gym_delta_mean": gym_delta,
        "gym_improved_count": gym_improved,
        "runs": runs,
    }


def _relative_gate(relative: dict[str, Any]) -> dict[str, Any]:
    subset = relative.get("feature_subset_audits", {}).get("fair_observed_behavior", {})
    fair_gate = subset.get("gate", {}) or {}
    raw_next = relative.get("feature_subset_audits", {}).get("latent_next_state_target_drift", {})
    summaries = subset.get("summaries", {}) or {}
    runs: dict[str, Any] = {}
    for name in ("prior_full", "prior_q90"):
        item = summaries.get(name, {}) or {}
        gate = fair_gate.get(name, {}) or {}
        runs[name] = {
            "pass": bool(gate.get("pass", False)),
            "complexity_mean": item.get("complexity_mean"),
            "mean_gap_vs_gym": item.get("mean_gap_vs_gym"),
            "ks_vs_gym": item.get("ks_vs_gym"),
            "low_mid_high": [
                item.get("low_complex_count"),
                item.get("mid_complex_count"),
                item.get("high_complex_count"),
            ],
            "improved_low_mid_high": [
                item.get("improved_low_complex_count"),
                item.get("improved_mid_complex_count"),
                item.get("improved_high_complex_count"),
            ],
            "gate_bits": gate,
        }
    return {
        "pass": bool(all(item["pass"] for item in runs.values())),
        "runs": runs,
        "raw_next_state_target_diagnostic_gate": raw_next.get("gate", {}),
        "note": (
            "next_state target drift is reported but excluded from the fair gate "
            "because Gym next_state and prior next_state are not isomorphic in the runner."
        ),
    }


def _pack_fit_gate(report: dict[str, Any]) -> dict[str, Any]:
    cases = [
        {"case": item.get("case"), "hard_pass": bool(item.get("hard_pass", False))}
        for item in report.get("cases", [])
    ]
    return {
        "pass": bool(report.get("hard_pass", False) and cases and all(item["hard_pass"] for item in cases)),
        "cases": cases,
    }


def _env_property_gate(report: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for item in report.get("cases", []):
        terminal = item.get("pack_property", {}).get("terminal_reset", {})
        reward = item.get("pack_property", {}).get("reward", {})
        cases.append(
            {
                "case": item.get("case"),
                "hard_pass": bool(item.get("hard_pass", False)),
                "n_envs": item.get("n_envs"),
                "n_steps": item.get("n_steps"),
                "terminal_target_mean": (terminal.get("target_meta", {}) or {}).get("mean"),
                "done_count_mean": (terminal.get("done_count_per_env", {}) or {}).get("mean"),
                "training_reward_std_mean": (
                    reward.get("training_reward_per_env_std", {}) or {}
                ).get("mean"),
                "raw_reward_std_mean": (reward.get("raw_reward_per_env_std", {}) or {}).get("mean"),
            }
        )
    return {
        "pass": bool(report.get("hard_pass", False) and cases and all(item["hard_pass"] for item in cases)),
        "cases": cases,
    }


def _ppo_gate(core: dict[str, Any], real: dict[str, Any], milestone: dict[str, Any]) -> dict[str, Any]:
    milestone_checks = milestone.get("checks", {}) or {}
    return {
        "pass": bool(
            core.get("passed", False)
            and real.get("passed", False)
            and milestone.get("hard_pass", False)
            and all(bool(item.get("report_hard_pass", False)) for item in milestone_checks.values())
        ),
        "core_passed": bool(core.get("passed", False)),
        "core_max_abs_diff": core.get("max_abs_diff"),
        "real_rollout_passed": bool(real.get("passed", False)),
        "real_rollout_max_abs_diff": real.get("max_abs_diff"),
        "prior_milestone_hard_pass": bool(milestone.get("hard_pass", False)),
        "prior_milestone_checks": {
            name: {
                "report_hard_pass": bool(item.get("report_hard_pass", False)),
                "generator_parity_hard_pass": bool(item.get("generator_parity_hard_pass", False)),
                "prior_pack_ppo_update_hard_pass": bool(
                    item.get("prior_pack_ppo_update_hard_pass", False)
                ),
            }
            for name, item in sorted(milestone_checks.items())
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# Profile Longrun Preflight Gate",
        "",
        f"- overall longrun ready: `{report['overall_longrun_ready']}`",
        "",
        "| gate | pass | read |",
        "| --- | ---: | --- |",
    ]
    rows = [
        (
            "selector/mode/terminal",
            checks["selector_mode_terminal"]["pass"],
            "mode counts, joint profile cap, terminal bins, opt proxy",
        ),
        (
            "optimizability",
            checks["optimizability"]["pass"],
            "abs delta ratio and improved-count against Gym",
        ),
        (
            "spectral rank",
            checks["spectral_rank"]["pass"],
            "improved behavior effective rank not below Gym threshold",
        ),
        (
            "fair observed complexity",
            checks["fair_observed_complexity"]["pass"],
            "Gym-calibrated obs/action/reward/done complexity distribution",
        ),
        (
            "pack-fit parity",
            checks["pack_fit_parity"]["pass"],
            "same h/seed pack collection equals fit_model bridge",
        ),
        (
            "env property parity",
            checks["env_property_parity"]["pass"],
            "terminal/reward/done metadata and rollout properties",
        ),
        (
            "PPO/M3 parity",
            checks["ppo_m3_parity"]["pass"],
            "GAE/update/VecNormalize/official PPO parity",
        ),
    ]
    for name, passed, read in rows:
        lines.append(f"| `{name}` | `{passed}` | {read} |")

    lines.extend(["", "## Optimizability", "", "| run | delta | delta/Gym | improved | e-rank/Gym | pass |"])
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    opt = checks["optimizability"]
    for name, item in opt["runs"].items():
        ratio = item["delta_ratio_vs_gym"]
        lines.append(
            f"| `{name}` | {item['delta_mean']:+.2f} | "
            f"{ratio:.2f} | {item['improved_count']}/{opt['gym_improved_count']} | "
            f"{item['improved_effective_rank_vs_gym']:.2f} | `{item['pass']}` |"
        )

    lines.extend(["", "## Terminal And Modes", "", "| rule | terminal bins | opt proxy | modes | active cells | dominant | pass |"])
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for rule, item in checks["selector_mode_terminal"]["rules"].items():
        modes = ", ".join(f"{key}:{value}" for key, value in item["mode_counts"].items())
        lines.append(
            f"| `{rule}` | `{item['terminal']['bins_0_20_40_60_80']}` | "
            f"{item['optimizability_proxy_count']}/{item['optimizability_proxy_min']} | "
            f"`{modes}` | {item['joint_profile_active_cells']} | "
            f"{item['joint_profile_dominant_rate']:.3f} | `{item['pass']}` |"
        )

    lines.extend(["", "## Fair Observed Complexity", "", "| run | mean | gap | KS | low/mid/high | improved low/mid/high | pass |"])
    lines.append("| --- | ---: | ---: | ---: | --- | --- | ---: |")
    for name, item in checks["fair_observed_complexity"]["runs"].items():
        lines.append(
            f"| `{name}` | {item['complexity_mean']:.3f} | "
            f"{item['mean_gap_vs_gym']:+.3f} | {item['ks_vs_gym']:.3f} | "
            f"`{item['low_mid_high']}` | `{item['improved_low_mid_high']}` | `{item['pass']}` |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- A `False` overall result blocks long fit_model submission; it does not imply the selector should be patched blindly.",
            "- `next_state` raw complexity remains diagnostic only because the current runner uses non-isomorphic next_state targets for Gym and prior.",
            "- The current fair observed complexity failure points to source/generator observed-behavior support, not improved-count or terminal skew.",
            "",
            "## Files",
            "",
            f"- JSON: `{report['outputs']['json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    selector = _read_json(args.selector_report)
    selector_rows = _read_csv(args.selector_csv)
    spectral = _read_json(args.spectral_json)
    relative = _read_json(args.relative_json)
    pack_fit = _read_json(args.pack_fit_parity_json)
    env_property = _read_json(args.env_property_parity_json)
    ppo_core = _read_json(args.ppo_core_json)
    ppo_real = _read_json(args.ppo_real_rollout_json)
    prior_milestone = _read_json(args.prior_milestone_json)

    spectral_pass = bool(spectral.get("hard_read", {}).get("spectral_dimension_gate_pass", False))
    checks = {
        "selector_mode_terminal": _selector_gate(selector, selector_rows),
        "optimizability": _optimizability_gate(spectral),
        "spectral_rank": _pass("spectral_rank", spectral_pass, spectral.get("comparisons", {})),
        "fair_observed_complexity": _relative_gate(relative),
        "pack_fit_parity": _pack_fit_gate(pack_fit),
        "env_property_parity": _env_property_gate(env_property),
        "ppo_m3_parity": _ppo_gate(ppo_core, ppo_real, prior_milestone),
    }
    overall = bool(all(bool(item.get("pass", False)) for item in checks.values()))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile_longrun_preflight_gate.json"
    md_path = out_dir / "profile_longrun_preflight_gate.md"
    report = {
        "analysis_entry": "phase2_profile_longrun_preflight_gate",
        "contract": {
            "read_only": True,
            "does_not_sample_envs": True,
            "does_not_train": True,
            "does_not_modify_selector": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "same_prior_pack_evidence_only": True,
        },
        "inputs": {
            "selector_report": str(Path(args.selector_report).expanduser().resolve()),
            "selector_csv": str(Path(args.selector_csv).expanduser().resolve()),
            "spectral_json": str(Path(args.spectral_json).expanduser().resolve()),
            "relative_json": str(Path(args.relative_json).expanduser().resolve()),
            "pack_fit_parity_json": str(Path(args.pack_fit_parity_json).expanduser().resolve()),
            "env_property_parity_json": str(Path(args.env_property_parity_json).expanduser().resolve()),
            "ppo_core_json": str(Path(args.ppo_core_json).expanduser().resolve()),
            "ppo_real_rollout_json": str(Path(args.ppo_real_rollout_json).expanduser().resolve()),
            "prior_milestone_json": str(Path(args.prior_milestone_json).expanduser().resolve()),
        },
        "overall_longrun_ready": overall,
        "checks": checks,
        "outputs": {"json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"overall_longrun_ready": overall, "json": str(json_path), "md": str(md_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-report", default=str(DEFAULT_SELECTOR_REPORT))
    parser.add_argument("--selector-csv", default=str(DEFAULT_SELECTOR_CSV))
    parser.add_argument("--spectral-json", default=str(DEFAULT_SPECTRAL_JSON))
    parser.add_argument("--relative-json", default=str(DEFAULT_RELATIVE_JSON))
    parser.add_argument("--pack-fit-parity-json", default=str(DEFAULT_PACK_FIT_PARITY_JSON))
    parser.add_argument("--env-property-parity-json", default=str(DEFAULT_ENV_PROPERTY_PARITY_JSON))
    parser.add_argument("--ppo-core-json", default=str(DEFAULT_PPO_CORE_JSON))
    parser.add_argument("--ppo-real-rollout-json", default=str(DEFAULT_PPO_REAL_ROLLOUT_JSON))
    parser.add_argument("--prior-milestone-json", default=str(DEFAULT_PRIOR_MILESTONE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
