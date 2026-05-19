#!/usr/bin/env python
"""Audit the unprofiled non-pendulum platform pressure point.

Read-only diagnostic for the v2o profile/prior-pack line.  This does not train,
sample environments, or mutate PPO / exact-SCM semantics.  It formalizes the
current negative pressure cell:

* not effective pendulum support,
* not low-energy,
* not state-conditioned energy injection,
* not strict feedback.

The cell is intentionally a pressure label, not a selector quota.  Its job is to
show whether a proposed source/generator change removes the same failure mode
that the optimizability sidecars observe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2o_unprofiled_platform_audit_0515"
DEFAULT_SOURCE_ANNOTATED_CSV = (
    ART / "phase2_profile_v2o_source_badgroup_audit_0515/full_source384_with_mode_badgroup.csv"
)
DEFAULT_V2O_SELECTED_CSV = (
    ART
    / "phase2_profile_band_selector_v2o_reductive_independent_uniform_terminal_seed9730_0515"
    / "selected_profile_band_prior_envs.csv"
)

DEFAULT_RUN_SPECS: tuple[tuple[str, Path, Path, str], ...] = (
    (
        "base_full_u4",
        DEFAULT_V2O_SELECTED_CSV,
        ART / "phase2_profile_v2o_currentmaint_prior_full_optparity_u4_b512_advnorm_takeown_0515",
        "gym_full_obs_action_range",
    ),
    (
        "base_q90_u4",
        DEFAULT_V2O_SELECTED_CSV,
        ART / "phase2_profile_v2o_currentmaint_gym_prior_q90_optparity_u4_b512_advnorm_takeown_0515",
        "gym_q90_obs_action_range",
    ),
    (
        "pend_bias0_u2",
        ART
        / "phase2_profile_v2o_pendulum_costsurface_bias0_cf_0515"
        / "selected_profile_band_prior_envs_pendulum_costsurface_bias0_cf.csv",
        ART / "phase2_profile_v2o_pendulum_costsurface_bias0_cf_u2_b512_advnorm_takeown_0515",
        "gym_full_obs_action_range",
    ),
    (
        "pend_bias0_q90_u2",
        ART
        / "phase2_profile_v2o_pendulum_costsurface_bias0_q90_cf_0515"
        / "selected_profile_band_prior_envs_pendulum_costsurface_bias0_q90_cf.csv",
        ART / "phase2_profile_v2o_pendulum_costsurface_bias0_q90_cf_u2_b512_advnorm_takeown_0515",
        "gym_q90_obs_action_range",
    ),
    (
        "nonpend_relu_u2",
        ART
        / "phase2_profile_v2o_pend_bias0_nonpend_relu_cf_0515"
        / "selected_profile_band_prior_envs_pend_bias0_nonpend_relu_cf.csv",
        ART / "phase2_profile_v2o_pend_bias0_nonpend_relu_cf_u2_b512_advnorm_takeown_0515",
        "gym_full_obs_action_range",
    ),
    (
        "nonpend_reward_bias_zero_u2",
        ART
        / "phase2_profile_v2o_pend_bias0_nonpend_rewardbiaszero_cf_0515"
        / "selected_profile_band_prior_envs_pend_bias0_nonpend_rewardbiaszero_cf.csv",
        ART
        / "phase2_profile_v2o_pend_bias0_nonpend_rewardbiaszero_cf_u2_b512_advnorm_takeown_0515",
        "gym_full_obs_action_range",
    ),
    (
        "badgroup_replacement_u2",
        ART
        / "phase2_profile_v2o_bias0_badgroup_replacement_cf_0515"
        / "selected_profile_band_prior_envs_bias0_badgroup_replacement_cf.csv",
        ART / "phase2_profile_v2o_bias0_badgroup_replacement_cf_u2_b512_advnorm_takeown_0515",
        "gym_full_obs_action_range",
    ),
)

STRUCTURE_KEYS = (
    "terminal_reset_count_target",
    "action_dim",
    "obs_dim",
    "state_dim",
    "noise_dim",
    "zero_pad_dim",
    "balanced_reward_state_input_gain_fraction",
    "balanced_reward_action_input_gain_fraction",
    "balanced_reward_noise_input_gain_fraction",
    "balanced_reward_state_to_action_gain_ratio",
)

MODE_KEYS = (
    "pendulum_axis_support",
    "low_energy_not_ctrl_only",
    "state_conditioned_energy_injection",
    "strict_feedback_candidate",
    "locomotion_witness",
    "sign_or_phase_sensitive",
    "feedback_near_best_support_set",
)


class RunSpec(NamedTuple):
    name: str
    selected_csv: Path
    run_dir: Path
    rule: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _last_jsonl(path: str | Path) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    if not Path(path).exists():
        return None
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                last = json.loads(text)
    return last


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "none", "nan"}:
        return False
    return bool(value)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _stats(values: list[float]) -> dict[str, Any]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0}
    return {
        "n": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "q25": float(np.quantile(clean, 0.25)),
        "q50": float(np.quantile(clean, 0.50)),
        "q75": float(np.quantile(clean, 0.75)),
        "max": float(np.max(clean)),
    }


def _rank_key(row: dict[str, Any]) -> int:
    return int(float(row.get("profile_band_rank", row.get("env_idx", 0))))


def _source_key(row: dict[str, Any]) -> tuple[str, int] | None:
    raw = row.get("_source_index")
    rule = row.get("candidate_rule") or row.get("rule")
    if raw in (None, "") or not rule:
        return None
    return str(rule), int(float(raw))


def is_unprofiled_platform(row: dict[str, Any]) -> bool:
    """Current v2o pressure cell.

    ``pendulum_short_gap`` is deliberately not counted as support here: the
    full failure used effective axis support, and short-gap alone did not
    protect optimizability.
    """

    if "unsafe_nonpend_triple_false" in row:
        return _safe_bool(row.get("unsafe_nonpend_triple_false"))
    pendulum_supported = _safe_bool(row.get("pendulum_axis_support")) or _safe_bool(
        row.get("pendulum_axis_supported_sustained_settle")
    )
    return (
        not pendulum_supported
        and not _safe_bool(row.get("low_energy_not_ctrl_only"))
        and not _safe_bool(row.get("state_conditioned_energy_injection"))
        and not _safe_bool(row.get("strict_feedback_candidate"))
    )


def _selected_rows_for_rule(path: str | Path, rule: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    filtered = [row for row in rows if str(row.get("candidate_rule") or row.get("rule")) == rule]
    return filtered or rows


def _group_summary(rows: list[dict[str, Any]], *, label_key: str = "unprofiled_platform") -> dict[str, Any]:
    n = len(rows)
    flagged = [row for row in rows if _safe_bool(row.get(label_key))]
    out: dict[str, Any] = {
        "n": n,
        "unprofiled_platform_count": len(flagged),
        "unprofiled_platform_rate": float(len(flagged) / n) if n else 0.0,
    }
    for key in MODE_KEYS:
        vals = [_safe_bool(row.get(key)) for row in rows if key in row]
        if vals:
            out[f"{key}_count"] = int(sum(vals))
            out[f"{key}_rate"] = float(sum(vals) / len(vals))
    cells = Counter(str(row.get("cell", "")) for row in rows if row.get("cell", "") != "")
    if cells:
        out["cell_top"] = [
            {"cell": cell, "count": count, "rate": float(count / n)}
            for cell, count in cells.most_common(8)
        ]
    return out


def _structure_delta(rows: list[dict[str, Any]], *, label_key: str = "unprofiled_platform") -> list[dict[str, Any]]:
    labels = np.asarray([1.0 if _safe_bool(row.get(label_key)) else 0.0 for row in rows], dtype=np.float64)
    out = []
    for key in STRUCTURE_KEYS:
        vals = np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
        finite = np.isfinite(vals)
        if int(np.sum(finite)) < 3 or np.std(vals[finite]) <= 1e-12:
            continue
        yes = vals[finite & (labels > 0.5)]
        no = vals[finite & (labels <= 0.5)]
        if yes.size == 0 or no.size == 0:
            continue
        pooled_std = float(np.std(vals[finite]))
        mean_gap = float(np.mean(yes) - np.mean(no))
        denom = pooled_std if pooled_std > 1e-12 else 1.0
        severity = "low"
        if abs(mean_gap / denom) >= 1.0:
            severity = "high"
        elif abs(mean_gap / denom) >= 0.5:
            severity = "medium"
        out.append(
            {
                "key": key,
                "unprofiled_mean": float(np.mean(yes)),
                "other_mean": float(np.mean(no)),
                "pooled_std": pooled_std,
                "standardized_mean_gap": float(mean_gap / denom),
                "severity": severity,
            }
        )
    return sorted(out, key=lambda row: abs(float(row["standardized_mean_gap"])), reverse=True)


def _load_sidecar_pair(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sidecar_dir = Path(run_dir).expanduser().resolve() / "semantic_sidecars"
    files = sorted(sidecar_dir.glob("prior_update_*_envs.jsonl"))
    if len(files) < 2:
        raise FileNotFoundError(f"need at least two prior semantic sidecars in {sidecar_dir}")
    return _read_jsonl(files[0]), _read_jsonl(files[-1])


def _ppo_health(run_dir: str | Path) -> dict[str, Any]:
    last = _last_jsonl(Path(run_dir).expanduser().resolve() / "prior_progress.jsonl") or {}
    update = last.get("update") or {}
    logger = update.get("logger") or {}
    param_delta = update.get("param_delta") or {}
    return {
        "approx_kl": logger.get("train/approx_kl"),
        "clip_fraction": logger.get("train/clip_fraction"),
        "explained_variance": logger.get("train/explained_variance"),
        "param_delta_l2": param_delta.get("param_delta_l2"),
        "param_delta_absmax": param_delta.get("param_delta_absmax"),
        "update_wall_time_sec": logger.get("train/update_wall_time_sec"),
        "outer_batches": logger.get("train/outer_batches"),
        "subbatches": logger.get("train/subbatches"),
        "exception": update.get("exception"),
    }


def _run_env_rows(spec: RunSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = sorted(_selected_rows_for_rule(spec.selected_csv, spec.rule), key=_rank_key)
    start_rows, end_rows = _load_sidecar_pair(spec.run_dir)
    start_by_idx = {int(row["env_idx"]): row for row in start_rows}
    end_by_idx = {int(row["env_idx"]): row for row in end_rows}
    env_rows = []
    for env_idx, selected_row in enumerate(selected):
        start = start_by_idx.get(env_idx)
        end = end_by_idx.get(env_idx)
        if start is None or end is None:
            continue
        train0 = _safe_float(start.get("training_reward_sum"))
        train1 = _safe_float(end.get("training_reward_sum"))
        raw0 = _safe_float(start.get("raw_reward_sum"))
        raw1 = _safe_float(end.get("raw_reward_sum"))
        row = {
            "run": spec.name,
            "rule": spec.rule,
            "env_idx": env_idx,
            "source_index": selected_row.get("_source_index"),
            "profile_band_rank": selected_row.get("profile_band_rank", env_idx),
            "unprofiled_platform": is_unprofiled_platform(selected_row),
            "unprofiled_platform_no_locomotion": is_unprofiled_platform(selected_row)
            and not _safe_bool(selected_row.get("locomotion_witness"))
            and not _safe_bool(selected_row.get("feedback_near_best_support_set")),
            "sign_only_unprofiled_platform": is_unprofiled_platform(selected_row)
            and _safe_bool(selected_row.get("sign_or_phase_sensitive"))
            and not _safe_bool(selected_row.get("locomotion_witness")),
            "train_reward_sum_start": train0,
            "train_reward_sum_end": train1,
            "train_reward_sum_delta": train1 - train0,
            "train_improved": train1 - train0 > 0.0,
            "raw_reward_sum_start": raw0,
            "raw_reward_sum_end": raw1,
            "raw_reward_sum_delta": raw1 - raw0,
            "raw_improved": raw1 - raw0 > 0.0,
            "raw_reward_std_start": _safe_float(start.get("raw_reward_std")),
            "raw_reward_std_end": _safe_float(end.get("raw_reward_std")),
            "reward_env_positive_fraction_start": _safe_float(
                start.get("reward_env_positive_fraction")
            ),
            "reward_env_positive_fraction_end": _safe_float(end.get("reward_env_positive_fraction")),
        }
        for key in MODE_KEYS + STRUCTURE_KEYS:
            if key in selected_row:
                row[key] = selected_row.get(key)
        env_rows.append(row)
    return env_rows, _ppo_health(spec.run_dir)


def _run_group_rows(env_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "all": lambda row: True,
        "pendulum_axis_support": lambda row: _safe_bool(row.get("pendulum_axis_support")),
        "unprofiled_platform": lambda row: _safe_bool(row.get("unprofiled_platform")),
        "unprofiled_platform_no_locomotion": lambda row: _safe_bool(
            row.get("unprofiled_platform_no_locomotion")
        ),
        "sign_only_unprofiled_platform": lambda row: _safe_bool(
            row.get("sign_only_unprofiled_platform")
        ),
        "supported_nonpend": lambda row: (
            not _safe_bool(row.get("pendulum_axis_support"))
            and not _safe_bool(row.get("unprofiled_platform"))
        ),
    }
    rows = []
    by_run = sorted({str(row["run"]) for row in env_rows})
    for run in by_run:
        scoped = [row for row in env_rows if str(row["run"]) == run]
        for group, predicate in groups.items():
            sub = [row for row in scoped if predicate(row)]
            if not sub:
                continue
            deltas = [_safe_float(row.get("train_reward_sum_delta")) for row in sub]
            raw_deltas = [_safe_float(row.get("raw_reward_sum_delta")) for row in sub]
            train0 = [_safe_float(row.get("train_reward_sum_start")) for row in sub]
            stats = _stats(deltas)
            raw = _stats(raw_deltas)
            start = _stats(train0)
            rows.append(
                {
                    "run": run,
                    "group": group,
                    "n": len(sub),
                    "train_improved_count": int(sum(_safe_bool(row.get("train_improved")) for row in sub)),
                    "train_improved_rate": float(
                        sum(_safe_bool(row.get("train_improved")) for row in sub) / len(sub)
                    ),
                    "train_delta_mean": stats.get("mean"),
                    "train_delta_q25": stats.get("q25"),
                    "train_delta_q50": stats.get("q50"),
                    "train_delta_q75": stats.get("q75"),
                    "raw_improved_count": int(sum(_safe_bool(row.get("raw_improved")) for row in sub)),
                    "raw_improved_rate": float(
                        sum(_safe_bool(row.get("raw_improved")) for row in sub) / len(sub)
                    ),
                    "raw_delta_mean": raw.get("mean"),
                    "raw_delta_q25": raw.get("q25"),
                    "raw_delta_q50": raw.get("q50"),
                    "raw_delta_q75": raw.get("q75"),
                    "train_start_mean": start.get("mean"),
                    "delta_contribution_sum": float(np.nansum(deltas)),
                }
            )
    return rows


def _parse_run_spec(text: str) -> RunSpec:
    parts = text.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("run spec must be name:selected_csv:run_dir:rule")
    return RunSpec(parts[0], Path(parts[1]), Path(parts[2]), parts[3])


def build_report(
    *,
    source_annotated_csv: str | Path | None = DEFAULT_SOURCE_ANNOTATED_CSV,
    run_specs: list[RunSpec] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = run_specs or [RunSpec(name, selected, run_dir, rule) for name, selected, run_dir, rule in DEFAULT_RUN_SPECS]

    source_rows = _read_csv(source_annotated_csv) if source_annotated_csv and Path(source_annotated_csv).exists() else []
    for row in source_rows:
        row["unprofiled_platform"] = is_unprofiled_platform(row)

    selected_summaries = {}
    selected_structure = {}
    for spec in specs:
        selected = _selected_rows_for_rule(spec.selected_csv, spec.rule)
        for row in selected:
            row["unprofiled_platform"] = is_unprofiled_platform(row)
        selected_summaries[spec.name] = _group_summary(selected)
        selected_structure[spec.name] = _structure_delta(selected)

    all_env_rows = []
    ppo_health = {}
    for spec in specs:
        rows, health = _run_env_rows(spec)
        all_env_rows.extend(rows)
        ppo_health[spec.name] = health
    group_rows = _run_group_rows(all_env_rows)

    source_summary: dict[str, Any] | None = None
    if source_rows:
        source_summary = _group_summary(source_rows)
        source_summary["by_rule"] = {
            rule: _group_summary([row for row in source_rows if str(row.get("rule")) == rule])
            for rule in sorted({str(row.get("rule")) for row in source_rows})
        }
        source_summary["structure_pressure"] = _structure_delta(source_rows)

    report = {
        "contract": {
            "read_only": True,
            "does_not_train": True,
            "does_not_modify_ppo_m3_or_exact_scm": True,
            "negative_pressure_label_not_selector_quota": True,
            "delta_metrics_are_guards_not_objectives": True,
            "definition_uses_pendulum_axis_support_not_short_gap": True,
        },
        "definition": {
            "unprofiled_platform": (
                "not pendulum_axis_support and not low_energy_not_ctrl_only and not "
                "state_conditioned_energy_injection and not strict_feedback_candidate"
            ),
            "why_not_pendulum_short_gap": (
                "short-gap alone did not protect full optimizability; the observed failure "
                "used effective pendulum axis support as the support boundary"
            ),
            "interpretation": (
                "This is a negative pressure cell showing insufficient profile definition or "
                "source-distribution support.  It should drive source/generator fixes, not "
                "become a hidden terminal/dim selector code."
            ),
        },
        "source_pool": source_summary,
        "selected_sets": selected_summaries,
        "selected_structure_pressure": selected_structure,
        "ppo_health": ppo_health,
        "run_group_metrics": group_rows,
        "decision": {
            "current_root_cause_read": (
                "Pendulum is addressed by the reward cost-surface/bias mechanism.  The remaining "
                "full failure is concentrated in the non-pendulum unprofiled-platform cell; "
                "ReLU/tanh/reward-output-bias counterfactuals do not remove it, while diagnostic "
                "replacement does."
            ),
            "next_mainline_fix": (
                "Define and reduce this cell at source/generator distribution level, then rerun "
                "full/q90 optimizability, structure independence, observed behavior, complexity/rank, "
                "and pack-to-fit_model parity."
            ),
        },
    }
    return _json_safe(report), _json_safe(group_rows), _json_safe(all_env_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotated-csv", default=str(DEFAULT_SOURCE_ANNOTATED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        help="Optional run spec: name:selected_csv:run_dir:rule. Repeatable.",
    )
    args = parser.parse_args()

    report, group_rows, env_rows = build_report(
        source_annotated_csv=args.source_annotated_csv,
        run_specs=args.run,
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile_unprofiled_platform_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(out_dir / "profile_unprofiled_platform_run_groups.csv", group_rows)
    _write_csv(out_dir / "profile_unprofiled_platform_env_rows.csv", env_rows)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
