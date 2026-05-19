#!/usr/bin/env python
"""Audit mode-profile coverage across milestone-2 and exploratory candidates.

Exploratory only.  This script does not sample environments, train PPO, or
modify the exact SCM / fit_model path.  Its purpose is to keep candidate
repairs profile-level: compare marginal mode rates, joint mode profiles, and
known subprofile bottlenecks before any generator change is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path("/home/chen/RLPFN/artifacts")
RULE_TO_SCOPE = {
    "gym_full_obs_action_range": "full",
    "gym_q90_obs_action_range": "q90",
}

BROAD_MODE_KEYS = (
    "locomotion_witness",
    "low_energy_not_ctrl_only",
    "sign_or_phase_sensitive",
    "state_conditioned_energy_injection",
)
FULL_MODE_KEYS = BROAD_MODE_KEYS + (
    "pendulum_like_primary_long_horizon",
    "pendulum_like_selector_hard_positive",
)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def _entropy(counter: Counter[tuple[bool, ...]]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counter.values():
        p = count / total
        out -= p * math.log2(p)
    return float(out)


def _scope_from_row(row: dict[str, Any]) -> str:
    if "scope" in row:
        return str(row["scope"])
    rule = str(row.get("rule", row.get("candidate_rule", "")))
    return RULE_TO_SCOPE.get(rule, rule)


def _profile_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    n = len(rows)
    counts: dict[str, int] = {key: sum(_safe_bool(row.get(key)) for row in rows) for key in keys}
    profile_counter: Counter[tuple[bool, ...]] = Counter(
        tuple(_safe_bool(row.get(key)) for key in keys) for row in rows
    )
    top_profiles = [
        {
            "profile": {key: value for key, value in zip(keys, profile)},
            "count": int(count),
            "rate": float(count / max(1, n)),
        }
        for profile, count in profile_counter.most_common(8)
    ]
    dominant_count = max(profile_counter.values()) if profile_counter else 0
    return {
        "n": int(n),
        "keys": list(keys),
        "mode_counts": {key: int(value) for key, value in counts.items()},
        "mode_rates": {key: float(value / max(1, n)) for key, value in counts.items()},
        "joint_profile_active_cells": int(len(profile_counter)),
        "joint_profile_entropy_bits": _entropy(profile_counter),
        "joint_profile_dominant_share": float(dominant_count / max(1, n)),
        "top_joint_profiles": top_profiles,
    }


def _split_by_scope(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"full": [], "q90": []}
    for row in rows:
        scope = _scope_from_row(row)
        if scope in out:
            out[scope].append(row)
    return out


def _candidate_summary(name: str, selected_csv: Path, smoke_json: Path | None) -> dict[str, Any]:
    rows = _read_csv(selected_csv)
    scopes = {}
    for scope, scoped_rows in _split_by_scope(rows).items():
        summary = _profile_summary(scoped_rows, BROAD_MODE_KEYS)
        if scoped_rows:
            summary["raw_reward_sum_delta_mean_in_selected_csv"] = _mean(
                [_safe_float(row.get("raw_reward_sum_delta")) for row in scoped_rows]
            )
        scopes[scope] = summary
    smoke = None
    if smoke_json and smoke_json.exists():
        data = _read_json(smoke_json)
        smoke = {
            "candidate_sufficient_for_smoke": data.get("overall", {}).get("candidate_sufficient_for_smoke"),
            "scopes": {},
        }
        for scope, body in data.get("scopes", {}).items():
            cand = body.get("candidate", {})
            raw = cand.get("raw_reward_sum_delta", {})
            diversity = body.get("diversity_guard", {})
            smoke["scopes"][scope] = {
                "raw_delta_mean": raw.get("mean"),
                "raw_delta_positive_fraction": raw.get("positive_fraction"),
                "diversity_guard_pass": diversity.get("pass"),
            }
    return {
        "name": name,
        "selected_csv": str(selected_csv),
        "smoke_json": str(smoke_json) if smoke_json else None,
        "scopes": scopes,
        "smoke": smoke,
    }


def _baseline_summary(per_env_csv: Path) -> dict[str, Any]:
    rows = _read_csv(per_env_csv)
    out: dict[str, Any] = {}
    for scope, scoped_rows in _split_by_scope(rows).items():
        broad = _profile_summary(scoped_rows, BROAD_MODE_KEYS)
        full = _profile_summary(scoped_rows, FULL_MODE_KEYS)
        broad["raw_reward_sum_delta_mean"] = _mean(
            [_safe_float(row.get("raw_reward_sum_delta")) for row in scoped_rows]
        )
        broad["raw_reward_sum_delta_positive_fraction"] = _mean(
            [1.0 if _safe_float(row.get("raw_reward_sum_delta")) > 0 else 0.0 for row in scoped_rows]
        )
        out[scope] = {"broad_profile": broad, "full_profile": full}
    return out


def _strict_summary(strict_report_json: Path) -> dict[str, Any]:
    data = _read_json(strict_report_json)
    return {
        "decision": data.get("decision"),
        "pool_summary": data.get("pool_summary"),
        "generated_lists": [
            {
                "name": item.get("name"),
                "n": item.get("n"),
                "csv": item.get("csv"),
                "summary": item.get("summary"),
            }
            for item in data.get("generated_lists", [])
        ],
        "frontier_upper_bound": data.get("frontier_upper_bound"),
    }


def _format_rate(value: Any) -> str:
    if value is None:
        return "na"
    try:
        f = float(value)
    except Exception:
        return str(value)
    return "na" if not math.isfinite(f) else f"{100.0 * f:.1f}%"


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Unified Profile Distribution Audit")
    lines.append("")
    lines.append("Exploratory-only audit.  No environment sampling, PPO training, exact SCM, or fit_model code path is modified.")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("- **Mode**: a trusted detector bit for one behavior family, e.g. locomotion witness or low-energy optimum.")
    lines.append("- **Joint profile**: the boolean vector formed by multiple modes for the same environment.  This catches domination by one easy combination even when each marginal rate looks acceptable.")
    lines.append("- **Subprofile**: a finer distribution inside a mode family.  The current important one is Pendulum-like strict feedback profile/gain/sign/feature.")
    lines.append("- **Guard**: a constraint that must not regress while increasing any target: state/reward/done diversity, raw post-pre delta rate, and protected mode coverage.")
    lines.append("")
    lines.append("## Current Milestone-2 Profile")
    lines.append("")
    lines.append("| scope | broad locomotion | low-energy | sign/phase | state-conditioned | Pendulum primary | Pendulum hard | broad dominant cell | full dominant cell | raw delta+ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for scope in ("full", "q90"):
        body = report["baseline"].get(scope, {})
        broad = body.get("broad_profile", {})
        full = body.get("full_profile", {})
        br = broad.get("mode_rates", {})
        fr = full.get("mode_rates", {})
        lines.append(
            "| {scope} | {loc} | {energy} | {sign} | {state} | {pend} | {hard} | {bdom} | {fdom} | {pos} |".format(
                scope=scope,
                loc=_format_rate(br.get("locomotion_witness")),
                energy=_format_rate(br.get("low_energy_not_ctrl_only")),
                sign=_format_rate(br.get("sign_or_phase_sensitive")),
                state=_format_rate(br.get("state_conditioned_energy_injection")),
                pend=_format_rate(fr.get("pendulum_like_primary_long_horizon")),
                hard=_format_rate(fr.get("pendulum_like_selector_hard_positive")),
                bdom=_format_rate(broad.get("joint_profile_dominant_share")),
                fdom=_format_rate(full.get("joint_profile_dominant_share")),
                pos=_format_rate(broad.get("raw_reward_sum_delta_positive_fraction")),
            )
        )
    lines.append("")
    lines.append("## Existing Candidate Profile/Smoke")
    lines.append("")
    lines.append("| candidate | scope | locomotion | low-energy | sign/phase | state-cond | dominant cell | raw delta mean | raw delta+ | diversity |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for cand in report["candidates"]:
        smoke_scopes = (cand.get("smoke") or {}).get("scopes", {})
        for scope in ("full", "q90"):
            body = cand["scopes"].get(scope, {})
            rates = body.get("mode_rates", {})
            smoke = smoke_scopes.get(scope, {})
            lines.append(
                "| {name} | {scope} | {loc} | {energy} | {sign} | {state} | {dom} | {mean} | {pos} | {div} |".format(
                    name=cand["name"],
                    scope=scope,
                    loc=_format_rate(rates.get("locomotion_witness")),
                    energy=_format_rate(rates.get("low_energy_not_ctrl_only")),
                    sign=_format_rate(rates.get("sign_or_phase_sensitive")),
                    state=_format_rate(rates.get("state_conditioned_energy_injection")),
                    dom=_format_rate(body.get("joint_profile_dominant_share")),
                    mean="na" if smoke.get("raw_delta_mean") is None else f"{float(smoke.get('raw_delta_mean')):+.3f}",
                    pos=_format_rate(smoke.get("raw_delta_positive_fraction")),
                    div=str(smoke.get("diversity_guard_pass")),
                )
            )
    lines.append("")
    lines.append("## Pendulum-Like Strict Subprofile")
    lines.append("")
    pool = report["strict_pendulum"]["pool_summary"]
    lines.append(
        f"- strict pool profile domination: `{pool['profile_dominance']['dominant']}` "
        f"{_format_rate(pool['profile_dominance']['dominant_share'])}; "
        f"gain domination: `{pool['gain_dominance']['dominant']}` "
        f"{_format_rate(pool['gain_dominance']['dominant_share'])}."
    )
    for item in report["strict_pendulum"]["generated_lists"]:
        summary = item["summary"]
        lines.append(
            f"- best-effort `{item['name']}`: profile dominant "
            f"{summary['profile_dominance']['dominant']} "
            f"{_format_rate(summary['profile_dominance']['dominant_share'])}; "
            f"gain dominant {summary['gain_dominance']['dominant']} "
            f"{_format_rate(summary['gain_dominance']['dominant_share'])}."
        )
    lines.append("")
    lines.append("## Distribution Adjustment Rule")
    lines.append("")
    lines.append("The next candidate should be profile-balanced, not single-mode maximized:")
    lines.append("")
    lines.append("1. Keep milestone-2 as the background generator and keep full/q90 as separate strata.")
    lines.append("2. Use locomotion as a raised marginal target only inside a joint-profile cap, because it is the clearest scarce locomotion/Gym mode.")
    lines.append("3. Keep low-energy, sign/phase, and state-conditioned rates inside bands; do not let any one joint profile dominate.")
    lines.append("4. Do not blindly increase Pendulum-like strict count; first fix its subprofile/gain domination.  Selection-only cannot fully solve this for n64.")
    lines.append("5. MountainCar-like upfront-investment is not yet an enforceable quota; it needs a detector before it can join the profile.")
    lines.append("")
    lines.append("## Proposed Guarded Profile Target")
    lines.append("")
    lines.append("- Full/q90 candidate size: 64 each.")
    lines.append("- Broad joint-profile dominant cell cap: target <= 25%, hard fail above 30%.")
    lines.append("- Locomotion witness: raise toward about 25-30%, but only if low-energy/sign/state-conditioned bands and smoke health remain non-regressed.")
    lines.append("- Low-energy/sign/state-conditioned: keep near milestone-2 bands, unless a future Gym-mode table proves a broader adjustment is necessary.")
    lines.append("- Pendulum strict subprofile: profile constant should not dominate above roughly 60% in the strict subset; gain=2.0 should not dominate above roughly 60%.  This is a subprofile guard, not a global quota increase.")
    lines.append("- MountainCar-like mode: add only after an upfront-control-investment detector is available.")
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append("The most defensible direction is a combined profile repair: preserve the two accepted milestone repairs, raise scarce locomotion under joint-profile guards, and repair Pendulum strict profile/gain domination without increasing its global fraction.  This avoids turning the prior into a one-mode generator while still addressing the clearest Gym-mode gaps.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACT_ROOT / "phase2_unified_profile_distribution_audit_0511"),
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = ARTIFACT_ROOT / "phase2_unified_gym_like_mode_profile_gpu0_0510/unified_gym_like_per_env_profile.csv"
    strict_json = (
        ARTIFACT_ROOT
        / "phase2_strict_profile_gain_balance_frontier_formal_0510/strict_profile_gain_balance_frontier_report.json"
    )
    candidates = [
        _candidate_summary(
            "locomotion_guarded",
            ARTIFACT_ROOT
            / "phase2_locomotion_band_guarded_candidate_tol0025_gated_n128_gpu0_0511/selected_locomotion_band_guarded_prior_envs.csv",
            ARTIFACT_ROOT
            / "phase2_locomotion_band_guarded_tol0025_smoke_summary_gpu0_0511/locomotion_guarded_smoke_summary.json",
        ),
        _candidate_summary(
            "energy_guarded",
            ARTIFACT_ROOT
            / "phase2_energy_band_guarded_candidate_tol0025_gated_n128_gpu0_0511/selected_energy_band_guarded_prior_envs.csv",
            ARTIFACT_ROOT
            / "phase2_energy_band_guarded_u4_smoke_summary_gpu0_0511/locomotion_guarded_smoke_summary.json",
        ),
        _candidate_summary(
            "feedback_guarded",
            ARTIFACT_ROOT
            / "phase2_feedback_band_guarded_candidate_tol0025_gated_n128_gpu0_0511/selected_feedback_band_guarded_prior_envs.csv",
            ARTIFACT_ROOT
            / "phase2_feedback_band_guarded_u4_smoke_summary_gpu0_0511/locomotion_guarded_smoke_summary.json",
        ),
        _candidate_summary(
            "locomotion_energy_guarded",
            ARTIFACT_ROOT
            / "phase2_locomotion_energy_band_guarded_candidate_tol0025_gated_n128_gpu0_0511/selected_locomotion_energy_band_guarded_prior_envs.csv",
            ARTIFACT_ROOT
            / "phase2_locomotion_energy_band_guarded_u4_smoke_summary_gpu0_0511/locomotion_guarded_smoke_summary.json",
        ),
        _candidate_summary(
            "joint_profile_guarded_locomotion",
            ARTIFACT_ROOT
            / "phase2_joint_profile_guarded_locomotion_candidate_gpu0_0511/selected_joint_profile_guarded_prior_envs.csv",
            None,
        ),
    ]
    report = {
        "analysis_entry": "phase2_unified_profile_distribution_audit",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "does_not_sample_new_envs": True,
            "does_not_train": True,
        },
        "definitions": {
            "broad_mode_keys": list(BROAD_MODE_KEYS),
            "full_mode_keys": list(FULL_MODE_KEYS),
            "profile": "boolean vector over mode keys for a single environment",
            "subprofile": "finer profile inside a mode family, currently Pendulum-like strict gain/profile/sign/feature",
        },
        "inputs": {
            "baseline_per_env_csv": str(baseline_csv),
            "strict_pendulum_report_json": str(strict_json),
        },
        "baseline": _baseline_summary(baseline_csv),
        "candidates": candidates,
        "strict_pendulum": _strict_summary(strict_json),
    }

    report_json = out_dir / "unified_profile_distribution_audit.json"
    report_md = out_dir / "unified_profile_distribution_audit.md"
    report_json.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {report_json}")
    print(f"wrote {report_md}")


if __name__ == "__main__":
    main()
