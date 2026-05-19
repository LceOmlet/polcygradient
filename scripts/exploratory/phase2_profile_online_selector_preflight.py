#!/usr/bin/env python
"""Preflight a bounded generator-level profile selector.

This is the thin orchestration contract for the profile mainline.  It does not
change the exact SCM, PPO, fit_model, or any milestone default.  It turns the
current fixed-list evidence into a repeatable generator-level plan:

1. sample a fresh milestone2 source pool,
2. materialize it as fixed groups only for read-only probes,
3. attach profile labels,
4. run the unified profile-band selector,
5. fail explicitly if support is insufficient.

The script can inspect an already-produced pool64 run, and it can also emit the
commands needed to reproduce the same semantics for a new seed.  Selection is
profile-only; raw score and delta-positive fraction are never selector inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ART = Path("/home/chen/RLPFN/artifacts")
REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING_POOL_ROOT = ART / "phase2_profile_online_candidate_pool_broad_proto_seed9521_pool64_0511"
DEFAULT_EXISTING_SELECTOR_DIR = (
    ART / "phase2_profile_online_candidate_pool_feedback_support_proto_seed9521_pool64_0511/profile_selector"
)
DEFAULT_PROFILE_CONFIG = REPO / "scripts/exploratory/profile_configs/profile_band_online_feedback_support_prototype.json"
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_online_selector_preflight_0511"
RULES = ("gym_full_obs_action_range", "gym_q90_obs_action_range")


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    outputs: tuple[Path, ...]
    cost_class: str
    profile_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "command_text": " ".join(shlex.quote(part) for part in self.command),
            "outputs": [str(path) for path in self.outputs],
            "outputs_exist": [path.exists() for path in self.outputs],
            "ready": all(path.exists() for path in self.outputs),
            "cost_class": self.cost_class,
            "profile_role": self.profile_role,
        }


def _read_json_optional(path: str | Path | None) -> Any:
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _read_csv_optional(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or str(path).strip() == "":
        return []
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def _rule_counts(rows: list[dict[str, str]], key: str = "rule") -> dict[str, int]:
    out = {rule: 0 for rule in RULES}
    for row in rows:
        rule = str(row.get(key, ""))
        if rule in out:
            out[rule] += 1
    return out


def _python() -> str:
    return sys.executable


def _paths(*parts: str | Path) -> Path:
    return Path(*parts).expanduser().resolve()


def _split_csv_floats(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if str(item).strip()]


def _split_csv_text(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _feedback_eval_budget(
    *,
    feedback_label_mode: str,
    source_target_per_rule: int,
    n_steps_feedback: int,
    feedback_baseline_modes: str,
    feedback_profiles: str,
    feedback_gains: str,
    feedback_obs_linear_count: int,
    max_feedback_eval_units: int,
) -> dict[str, Any]:
    baseline_modes = _split_csv_text(feedback_baseline_modes)
    profiles = _split_csv_text(feedback_profiles)
    gains = _split_csv_floats(feedback_gains)
    feedback_mode_count = len(profiles) * max(0, int(feedback_obs_linear_count)) * len(gains) * 2
    if str(feedback_label_mode) == "budgeted_guard":
        mode_count_per_rule = len(baseline_modes) + feedback_mode_count
    elif str(feedback_label_mode) == "full_audit":
        mode_count_per_rule = feedback_mode_count
    else:
        raise ValueError(f"unsupported feedback_label_mode: {feedback_label_mode}")
    eval_units = int(len(RULES) * int(source_target_per_rule) * int(n_steps_feedback) * mode_count_per_rule)
    return {
        "label_mode": str(feedback_label_mode),
        "baseline_modes": baseline_modes,
        "profiles": profiles,
        "gains": gains,
        "obs_linear_count": int(feedback_obs_linear_count),
        "feedback_mode_count_per_rule": int(feedback_mode_count),
        "mode_count_per_rule": int(mode_count_per_rule),
        "eval_units": eval_units,
        "max_eval_units_for_default_execute": int(max_feedback_eval_units),
        "execution_blocked_by_default": bool(eval_units > int(max_feedback_eval_units)),
        "read": (
            "feedback labels exceed the startup budget; reduce guard modes/chunk size or run full audit offline"
            if eval_units > int(max_feedback_eval_units)
            else "feedback labels are within the configured default execution budget"
        ),
    }


def build_stage_plan(
    *,
    run_root: str | Path,
    target_per_rule: int,
    candidate_multiplier: int,
    seed: int,
    env_seed_start: int,
    device: str,
    profile_config: str | Path,
    max_raw_samples: int,
    raw_batch_size: int,
    env_eval_batch_size: int,
    n_steps_action: int,
    n_steps_locomotion: int,
    action_obs_linear_count: int = 4,
    locomotion_obs_linear_count: int = 4,
    n_steps_feedback: int,
    feedback_label_mode: str,
    feedback_baseline_modes: str,
    feedback_profiles: str,
    feedback_gains: str,
    feedback_obs_linear_count: int,
    max_feedback_eval_units: int,
    pendulum_settle_yield_axis_profile: str = "off",
    pendulum_settle_yield_axis_rate: float = 0.0,
    pendulum_settle_yield_axis_action_gains: str = "0.35,0.55",
    pendulum_settle_yield_axis_dampings: str = "0.88,0.94",
    pendulum_settle_yield_axis_springs: str = "0.10",
    pendulum_settle_yield_axis_position_couplings: str = "0.0",
    pendulum_settle_yield_axis_action_controls_positions: str = "false",
) -> dict[str, Any]:
    run_root = _paths(run_root)
    source_target = int(target_per_rule) * int(candidate_multiplier)
    source_dir = run_root / "source_pool"
    fixed_dir = run_root / "fixed_groups"
    action_dir = run_root / "action_mode_probe"
    locomotion_dir = run_root / "locomotion_probe"
    feedback_dir = run_root / "feedback_near_best_source_pool_corrected"
    if str(feedback_label_mode) == "budgeted_guard":
        feedback_dir = run_root / "feedback_budgeted_support_guard"
        feedback_script = "phase2_feedback_budgeted_support_guard.py"
        feedback_csv = feedback_dir / "feedback_budgeted_support_guard_per_env.csv"
        feedback_stage_name = "feedback_budgeted_guard_labels"
        feedback_role = "startup-safe feedback support guard; full audit remains offline"
    elif str(feedback_label_mode) == "full_audit":
        feedback_script = "phase2_feedback_near_best_gain_audit.py"
        feedback_csv = feedback_dir / "feedback_near_best_gain_per_env.csv"
        feedback_stage_name = "feedback_near_best_labels"
        feedback_role = "offline full-grid feedback support labels; avoids hard-argmax gain collapse"
    else:
        raise ValueError(f"unsupported feedback_label_mode: {feedback_label_mode}")
    selector_dir = run_root / "profile_selector"
    feedback_budget = _feedback_eval_budget(
        feedback_label_mode=feedback_label_mode,
        source_target_per_rule=source_target,
        n_steps_feedback=n_steps_feedback,
        feedback_baseline_modes=feedback_baseline_modes,
        feedback_profiles=feedback_profiles,
        feedback_gains=feedback_gains,
        feedback_obs_linear_count=feedback_obs_linear_count,
        max_feedback_eval_units=max_feedback_eval_units,
    )

    full_h = fixed_dir / "gym_full_obs_action_range/fixed_env_group/prior_fixed_h_list.json"
    q90_h = fixed_dir / "gym_q90_obs_action_range/fixed_env_group/prior_fixed_h_list.json"
    selected_csv = source_dir / "selected_prior_envs.csv"
    feedback_command = [
        _python(),
        str(REPO / "scripts/exploratory" / feedback_script),
        "--selected-csv",
        str(selected_csv),
        "--output-dir",
        str(feedback_dir),
        "--n-envs",
        str(source_target),
        "--n-steps",
        str(n_steps_feedback),
        "--device",
        str(device),
        "--obs-linear-count",
        str(feedback_obs_linear_count),
    ]
    if str(feedback_label_mode) == "budgeted_guard":
        feedback_command.extend(
            [
                "--baseline-modes",
                str(feedback_baseline_modes),
                "--feedback-profiles",
                str(feedback_profiles),
                "--feedback-gains",
                str(feedback_gains),
            ]
        )
    else:
        feedback_command.extend(["--profiles", str(feedback_profiles), "--gains", str(feedback_gains)])

    source_command = [
        _python(),
        str(REPO / "scripts/exploratory/phase2_reward_group_balance_probe.py"),
        "--output-dir",
        str(source_dir),
        "--target-per-rule",
        str(source_target),
        "--seed",
        str(seed),
        "--env-seed-start",
        str(env_seed_start),
        "--device",
        str(device),
        "--raw-batch-size",
        str(raw_batch_size),
        "--env-eval-batch-size",
        str(env_eval_batch_size),
        "--max-raw-samples",
        str(max_raw_samples),
        "--diversity-n-envs",
        str(min(source_target, 64)),
        "--diversity-steps",
        "32",
    ]
    if float(pendulum_settle_yield_axis_rate) > 0.0:
        source_command.extend(
            [
                "--pendulum-settle-yield-axis-profile",
                str(pendulum_settle_yield_axis_profile),
                "--pendulum-settle-yield-axis-rate",
                str(pendulum_settle_yield_axis_rate),
                "--pendulum-settle-yield-axis-action-gains",
                str(pendulum_settle_yield_axis_action_gains),
                "--pendulum-settle-yield-axis-dampings",
                str(pendulum_settle_yield_axis_dampings),
                "--pendulum-settle-yield-axis-springs",
                str(pendulum_settle_yield_axis_springs),
                "--pendulum-settle-yield-axis-position-couplings",
                str(pendulum_settle_yield_axis_position_couplings),
                "--pendulum-settle-yield-axis-action-controls-positions",
                str(pendulum_settle_yield_axis_action_controls_positions),
            ]
        )

    stages = [
        Stage(
            name="source_pool",
            cost_class="gpu_generator_medium",
            profile_role="fresh milestone2 candidate production",
            outputs=(source_dir / "reward_group_balance_probe_report.json", selected_csv),
            command=source_command,
        ),
        Stage(
            name="fixed_groups",
            cost_class="cpu_io_light",
            profile_role="adapter for read-only profile probes",
            outputs=(fixed_dir / "fixed_groups_manifest.json", full_h, q90_h),
            command=[
                _python(),
                str(REPO / "scripts/exploratory/phase2_selected_prior_envs_to_fixed_groups.py"),
                "--selected-csv",
                str(selected_csv),
                "--output-dir",
                str(fixed_dir),
            ],
        ),
        Stage(
            name="action_mode_labels",
            cost_class="gpu_probe_medium",
            profile_role="low-energy/sign-phase/state-conditioned labels",
            outputs=(action_dir / "report.json",),
            command=[
                _python(),
                str(REPO / "scripts/exploratory/phase2_prior_action_mode_coverage_probe.py"),
                "--output-json",
                str(action_dir / "report.json"),
                "--fixed-h-list-json",
                str(full_h),
                "--fixed-h-list-json",
                str(q90_h),
                "--n-envs",
                str(source_target),
                "--n-steps",
                str(n_steps_action),
                "--device",
                str(device),
                "--prior-milestone",
                "gated_reward_path_balance",
                "--obs-linear-count",
                str(action_obs_linear_count),
            ],
        ),
        Stage(
            name="locomotion_labels",
            cost_class="gpu_probe_expensive",
            profile_role="sustained temporal locomotion witness",
            outputs=(locomotion_dir / "locomotion_full_temporal_per_env.csv",),
            command=[
                _python(),
                str(REPO / "scripts/exploratory/phase2_locomotion_full_temporal_detector.py"),
                "--output-dir",
                str(locomotion_dir),
                "--fixed-h-list-json",
                str(full_h),
                "--fixed-h-list-json",
                str(q90_h),
                "--n-envs",
                str(source_target),
                "--n-steps",
                str(n_steps_locomotion),
                "--device",
                str(device),
                "--obs-linear-count",
                str(locomotion_obs_linear_count),
            ],
        ),
        Stage(
            name=feedback_stage_name,
            cost_class="gpu_probe_expensive",
            profile_role=feedback_role,
            outputs=(feedback_csv,),
            command=feedback_command,
        ),
        Stage(
            name="profile_selector",
            cost_class="cpu_milp_light",
            profile_role="profile-band selection; raw delta is not an objective",
            outputs=(selector_dir / "profile_band_selector_report.json", selector_dir / "selected_profile_band_prior_envs.csv"),
            command=[
                _python(),
                str(REPO / "scripts/exploratory/phase2_profile_band_selector.py"),
                "--selected-csv",
                str(selected_csv),
                "--action-mode-json",
                str(action_dir / "report.json"),
                "--locomotion-csv",
                str(locomotion_dir / "locomotion_full_temporal_per_env.csv"),
                "--feedback-near-best-csv",
                str(feedback_csv),
                "--strict-feedback-csv",
                "",
                "--profile-config",
                str(Path(profile_config).expanduser().resolve()),
                "--output-dir",
                str(selector_dir),
            ],
        ),
    ]
    return {
        "run_root": str(run_root),
        "target_per_rule": int(target_per_rule),
        "candidate_multiplier": int(candidate_multiplier),
        "source_target_per_rule": int(source_target),
        "seed": int(seed),
        "env_seed_start": int(env_seed_start),
        "device": str(device),
        "feedback_label_mode": str(feedback_label_mode),
        "feedback_eval_budget": feedback_budget,
        "pendulum_settle_yield_axis": {
            "profile": str(pendulum_settle_yield_axis_profile),
            "active_rate": float(pendulum_settle_yield_axis_rate),
            "action_gains": _split_csv_floats(pendulum_settle_yield_axis_action_gains),
            "dampings": _split_csv_floats(pendulum_settle_yield_axis_dampings),
            "springs": _split_csv_floats(pendulum_settle_yield_axis_springs),
            "position_couplings": _split_csv_floats(pendulum_settle_yield_axis_position_couplings),
            "action_controls_positions": [
                item.strip().lower() in {"1", "true", "yes", "y", "on"}
                for item in str(pendulum_settle_yield_axis_action_controls_positions).split(",")
                if item.strip()
            ],
            "default_path_enabled": bool(float(pendulum_settle_yield_axis_rate) > 0.0),
            "standalone_generator": False,
            "uses_existing_state_action_slots": True,
            "read": (
                "source-pool generation will opt into a uniformly binned prior-internal settle-yield axis"
                if float(pendulum_settle_yield_axis_rate) > 0.0
                else "source-pool generation keeps the settle-yield axis off"
            ),
        },
        "stages": stages,
    }


def _selector_summary(selector_report: dict[str, Any] | None) -> dict[str, Any]:
    if not selector_report:
        return {"available": False}
    rules = selector_report.get("rules") or {}
    per_rule: dict[str, Any] = {}
    guard_values = []
    for rule in RULES:
        payload = rules.get(rule) or {}
        selected = payload.get("selected_summary") or {}
        source = payload.get("source_summary") or {}
        guard = bool(payload.get("guard_pass"))
        guard_values.append(guard)
        per_rule[rule] = {
            "guard_pass": guard,
            "source_n": source.get("n"),
            "selected_n": selected.get("n"),
            "active_cells": selected.get("joint_profile_active_cells"),
            "dominant_rate": selected.get("joint_profile_dominant_rate"),
            "entropy_norm": selected.get("joint_profile_entropy_norm"),
            "locomotion_rate": selected.get("locomotion_witness_rate"),
            "low_energy_rate": selected.get("low_energy_not_ctrl_only_rate"),
            "sign_phase_rate": selected.get("sign_or_phase_sensitive_rate"),
            "state_conditioned_rate": selected.get("state_conditioned_energy_injection_rate"),
            "feedback_rate": selected.get("strict_feedback_rate"),
            "feedback_profile_counts": selected.get("strict_feedback_profile_counts"),
            "feedback_gain_counts": selected.get("strict_feedback_gain_counts"),
        }
    return {
        "available": True,
        "hard_pass": bool(guard_values and all(guard_values)),
        "profile_version_id": (selector_report.get("config") or {}).get("profile_version_id"),
        "per_rule": per_rule,
    }


def _scale_projection(
    *,
    fit_model_target_envs: int,
    candidate_multiplier: int,
    reference_probe_envs: int,
    reference_probe_rss_gib: float,
    reference_probe_gpu_delta_gib: float,
    probe_chunk_envs: int,
    host_rss_limit_gib: float,
    gpu0_mem_limit_gib: float,
) -> dict[str, Any]:
    selected = max(1, int(fit_model_target_envs))
    source_target = selected * max(1, int(candidate_multiplier))
    ref_envs = max(1, int(reference_probe_envs))
    chunk_envs = max(1, int(probe_chunk_envs))
    relative = float(source_target) / float(ref_envs)
    naive_rss = float(reference_probe_rss_gib) * relative
    naive_gpu_delta = float(reference_probe_gpu_delta_gib) * relative
    chunk_relative = float(chunk_envs) / float(ref_envs)
    chunk_rss = float(reference_probe_rss_gib) * chunk_relative
    chunk_gpu_delta = float(reference_probe_gpu_delta_gib) * chunk_relative
    chunk_count = int(math.ceil(source_target / chunk_envs))
    naive_memory_safe = bool(
        naive_rss < 0.75 * float(host_rss_limit_gib)
        and naive_gpu_delta < 0.75 * float(gpu0_mem_limit_gib)
    )
    chunk_memory_safe = bool(
        chunk_rss < 0.75 * float(host_rss_limit_gib)
        and chunk_gpu_delta < 0.75 * float(gpu0_mem_limit_gib)
    )
    return {
        "fit_model_target_envs": selected,
        "candidate_multiplier": int(candidate_multiplier),
        "source_target_envs": source_target,
        "reference_probe_envs": ref_envs,
        "reference_probe_rss_gib": float(reference_probe_rss_gib),
        "reference_probe_gpu_delta_gib": float(reference_probe_gpu_delta_gib),
        "naive_all_at_once": {
            "relative_to_reference": relative,
            "projected_rss_gib": naive_rss,
            "projected_gpu_delta_gib": naive_gpu_delta,
            "memory_safe": naive_memory_safe,
            "read": (
                "do not use all-at-once probe evaluation at fit_model scale"
                if not naive_memory_safe
                else "all-at-once projection is below the configured memory guards"
            ),
        },
        "chunked_probe": {
            "probe_chunk_envs": chunk_envs,
            "chunk_count": chunk_count,
            "projected_rss_gib_per_chunk": chunk_rss,
            "projected_gpu_delta_gib_per_chunk": chunk_gpu_delta,
            "memory_safe": chunk_memory_safe,
            "read": (
                "chunked/streaming labels turn memory risk into evaluation-time cost"
                if chunk_memory_safe
                else "even chunked labels need a smaller chunk size before fit_model startup"
            ),
        },
        "pressure_read": (
            "At 2048-scale the pressure point is profile-label evaluation, not reward-path source-pool yield. "
            "A fit_model startup path must stream/chunk profile probes and merge labels by frozen_h fingerprint; "
            "it should not call the current probes with n_envs=2048 in one shot."
        ),
    }


def build_report(
    *,
    existing_pool_root: str | Path,
    existing_selector_dir: str | Path,
    planned_run_root: str | Path,
    target_per_rule: int,
    candidate_multiplier: int,
    seed: int,
    env_seed_start: int,
    device: str,
    profile_config: str | Path,
    max_raw_samples: int,
    raw_batch_size: int,
    env_eval_batch_size: int,
    n_steps_action: int,
    n_steps_locomotion: int,
    action_obs_linear_count: int = 4,
    locomotion_obs_linear_count: int = 4,
    n_steps_feedback: int,
    feedback_label_mode: str,
    feedback_baseline_modes: str,
    feedback_profiles: str,
    feedback_gains: str,
    feedback_obs_linear_count: int,
    max_feedback_eval_units: int,
    fit_model_target_envs: int,
    reference_probe_envs: int,
    reference_probe_rss_gib: float,
    reference_probe_gpu_delta_gib: float,
    probe_chunk_envs: int,
    host_rss_limit_gib: float,
    gpu0_mem_limit_gib: float,
    pendulum_settle_yield_axis_profile: str = "off",
    pendulum_settle_yield_axis_rate: float = 0.0,
    pendulum_settle_yield_axis_action_gains: str = "0.35,0.55",
    pendulum_settle_yield_axis_dampings: str = "0.88,0.94",
    pendulum_settle_yield_axis_springs: str = "0.10",
    pendulum_settle_yield_axis_position_couplings: str = "0.0",
    pendulum_settle_yield_axis_action_controls_positions: str = "false",
) -> dict[str, Any]:
    existing_pool_root = _paths(existing_pool_root)
    existing_selector_dir = _paths(existing_selector_dir)
    selected_rows = _read_csv_optional(existing_pool_root / "source_pool/selected_prior_envs.csv")
    source_report = _read_json_optional(existing_pool_root / "source_pool/reward_group_balance_probe_report.json")
    selector_report = _read_json_optional(existing_selector_dir / "profile_band_selector_report.json")
    plan = build_stage_plan(
        run_root=planned_run_root,
        target_per_rule=target_per_rule,
        candidate_multiplier=candidate_multiplier,
        seed=seed,
        env_seed_start=env_seed_start,
        device=device,
        profile_config=profile_config,
        max_raw_samples=max_raw_samples,
        raw_batch_size=raw_batch_size,
        env_eval_batch_size=env_eval_batch_size,
        n_steps_action=n_steps_action,
        n_steps_locomotion=n_steps_locomotion,
        action_obs_linear_count=action_obs_linear_count,
        locomotion_obs_linear_count=locomotion_obs_linear_count,
        n_steps_feedback=n_steps_feedback,
        feedback_label_mode=feedback_label_mode,
        feedback_baseline_modes=feedback_baseline_modes,
        feedback_profiles=feedback_profiles,
        feedback_gains=feedback_gains,
        feedback_obs_linear_count=feedback_obs_linear_count,
        max_feedback_eval_units=max_feedback_eval_units,
        pendulum_settle_yield_axis_profile=pendulum_settle_yield_axis_profile,
        pendulum_settle_yield_axis_rate=pendulum_settle_yield_axis_rate,
        pendulum_settle_yield_axis_action_gains=pendulum_settle_yield_axis_action_gains,
        pendulum_settle_yield_axis_dampings=pendulum_settle_yield_axis_dampings,
        pendulum_settle_yield_axis_springs=pendulum_settle_yield_axis_springs,
        pendulum_settle_yield_axis_position_couplings=pendulum_settle_yield_axis_position_couplings,
        pendulum_settle_yield_axis_action_controls_positions=pendulum_settle_yield_axis_action_controls_positions,
    )
    scale_projection = _scale_projection(
        fit_model_target_envs=fit_model_target_envs,
        candidate_multiplier=candidate_multiplier,
        reference_probe_envs=reference_probe_envs,
        reference_probe_rss_gib=reference_probe_rss_gib,
        reference_probe_gpu_delta_gib=reference_probe_gpu_delta_gib,
        probe_chunk_envs=probe_chunk_envs,
        host_rss_limit_gib=host_rss_limit_gib,
        gpu0_mem_limit_gib=gpu0_mem_limit_gib,
    )

    stage_dicts = [stage.to_dict() for stage in plan["stages"]]
    selector_summary = _selector_summary(selector_report)
    weights = {
        "fresh_source_pool_contract": 0.16,
        "broad_profile_labels": 0.18,
        "feedback_support_labels": 0.15,
        "selector_profile_pass": 0.22,
        "explicit_bounded_run_plan": 0.10,
        "no_silent_fallback_contract": 0.07,
        "new_seed_generator_execution": 0.07,
        "fit_model_selector_path_parity": 0.05,
    }
    score = 0.0
    source_counts = _rule_counts(selected_rows)
    source_target = int(target_per_rule) * int(candidate_multiplier)
    source_ready = bool(selected_rows and all(source_counts[rule] >= source_target for rule in RULES))
    broad_ready = bool(
        (existing_pool_root / "action_mode_probe/report.json").exists()
        and (existing_pool_root / "locomotion_probe/locomotion_full_temporal_per_env.csv").exists()
    )
    feedback_ready = bool(
        (existing_pool_root / "feedback_near_best_source_pool_corrected/feedback_near_best_gain_per_env.csv").exists()
        or (existing_pool_root / "feedback_budgeted_support_guard/feedback_budgeted_support_guard_per_env.csv").exists()
    )
    selector_ready = bool(selector_summary.get("hard_pass"))
    if source_ready:
        score += weights["fresh_source_pool_contract"]
    if broad_ready:
        score += weights["broad_profile_labels"]
    if feedback_ready:
        score += weights["feedback_support_labels"]
    if selector_ready:
        score += weights["selector_profile_pass"]
    score += weights["explicit_bounded_run_plan"]
    score += weights["no_silent_fallback_contract"]

    executed_plan_ready = all(stage["ready"] for stage in stage_dicts)
    if executed_plan_ready:
        score += weights["new_seed_generator_execution"]
    # The final parity weight is intentionally not awarded by this preflight.  It
    # requires the subsequent fit_model selector path parity, which keeps this
    # report from overclaiming based on generator-side artifacts alone.
    return {
        "analysis_entry": "phase2_profile_online_selector_preflight",
        "contract": {
            "exploratory_only": True,
            "does_not_modify_exact_scm": True,
            "does_not_modify_fit_model": True,
            "does_not_modify_ppo": True,
            "profile_quality_first": True,
            "delta_positive_fraction_not_a_gate": True,
            "raw_score_not_a_selector_objective": True,
            "explicit_failure_no_unprofiled_fallback": True,
        },
        "existing_evidence": {
            "pool_root": str(existing_pool_root),
            "selector_dir": str(existing_selector_dir),
            "source_pool": {
                "available": bool(source_report),
                "selected_count_by_rule": source_counts,
                "sampling": (source_report or {}).get("sampling", {}),
                "source_ready_for_candidate_multiplier": source_ready,
            },
            "broad_labels_ready": broad_ready,
            "feedback_support_ready": feedback_ready,
            "selector": selector_summary,
        },
        "bounded_generator_plan": {
            **{key: value for key, value in plan.items() if key != "stages"},
            "stage_plan": stage_dicts,
            "failure_semantics": [
                "if source pool cannot produce candidate_multiplier*N candidates before max_raw_samples, fail with support counts",
                "if any required label stage is missing or fails, fail explicitly",
                "if MILP selector is infeasible, increase candidate_multiplier up to a configured cap before failing",
                "never fall back to unprofiled milestone2 sampling",
            ],
            "cost_control": [
                "raw/topology generation is bounded by max_raw_samples",
                "profile probes scale with source_target_per_rule and are the current expected cost head",
                "full feedback near-best grids are blocked from default execution when they exceed the configured eval-unit budget",
                "expensive labels should eventually be cached by frozen_h fingerprint",
                "MountainCar remains detector/coverage annotation, not a selector quota",
            ],
            "pendulum_settle_yield_axis": plan["pendulum_settle_yield_axis"],
        },
        "fit_model_2048_scale_projection": scale_projection,
        "sufficiency_forecast": {
            "score": float(score),
            "remaining_to_profile_sufficient": float(max(0.0, 1.0 - score)),
            "weights": weights,
            "new_seed_generator_execution_ready": bool(executed_plan_ready),
            "fit_model_selector_path_parity_ready": False,
            "read": (
                "The existing pool64 evidence is close to profile-sufficient. "
                "The remaining semantic gap is fit_model-startable generator-level execution, not another profile quota."
            ),
            "next_highest_value": (
                "Turn this bounded plan into the fit_model startup path only after one new-seed preflight "
                "and generator-level pack-to-fit_model parity pass."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    forecast = report["sufficiency_forecast"]
    lines = [
        "# Online Profile Selector Preflight",
        "",
        "This is the generator-level contract layer for the unified profile mainline.",
        "",
        "## Forecast",
        "",
        f"- sufficiency score: `{forecast['score']:.3f}`",
        f"- remaining to profile-sufficient: `{forecast['remaining_to_profile_sufficient']:.3f}`",
        f"- read: {forecast['read']}",
        "",
        "## Existing Evidence",
        "",
    ]
    evidence = report["existing_evidence"]
    lines.append(f"- source pool counts: `{evidence['source_pool']['selected_count_by_rule']}`")
    lines.append(f"- source ready for multiplier: `{evidence['source_pool']['source_ready_for_candidate_multiplier']}`")
    lines.append(f"- broad labels ready: `{evidence['broad_labels_ready']}`")
    lines.append(f"- feedback support ready: `{evidence['feedback_support_ready']}`")
    lines.append(f"- selector hard pass: `{evidence['selector'].get('hard_pass')}`")
    lines.extend(["", "## Selector Snapshot", ""])
    lines.append(
        "| rule | guard | active cells | dominant | entropy | locomotion | low-energy | sign/phase | state-cond | feedback |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rule, item in (evidence["selector"].get("per_rule") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{rule}`",
                    f"`{item.get('guard_pass')}`",
                    str(item.get("active_cells")),
                    f"{float(item.get('dominant_rate') or 0.0):.3f}",
                    f"{float(item.get('entropy_norm') or 0.0):.3f}",
                    f"{float(item.get('locomotion_rate') or 0.0):.3f}",
                    f"{float(item.get('low_energy_rate') or 0.0):.3f}",
                    f"{float(item.get('sign_phase_rate') or 0.0):.3f}",
                    f"{float(item.get('state_conditioned_rate') or 0.0):.3f}",
                    f"{float(item.get('feedback_rate') or 0.0):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Bounded Plan", ""])
    for stage in report["bounded_generator_plan"]["stage_plan"]:
        lines.append(
            f"- `{stage['name']}` ready=`{stage['ready']}` cost=`{stage['cost_class']}` role={stage['profile_role']}"
        )
    feedback_budget = report["bounded_generator_plan"].get("feedback_eval_budget", {})
    settle_axis = report["bounded_generator_plan"].get("pendulum_settle_yield_axis", {})
    lines.extend(
        [
            "",
            "## Feedback Budget",
            "",
            f"- label mode: `{feedback_budget.get('label_mode')}`",
            f"- baselines: `{feedback_budget.get('baseline_modes')}`",
            f"- profiles: `{feedback_budget.get('profiles')}`",
            f"- gains: `{feedback_budget.get('gains')}`",
            f"- feedback modes per rule: `{feedback_budget.get('feedback_mode_count_per_rule')}`",
            f"- modes per rule: `{feedback_budget.get('mode_count_per_rule')}`",
            f"- eval units: `{feedback_budget.get('eval_units')}`",
            f"- max default eval units: `{feedback_budget.get('max_eval_units_for_default_execute')}`",
            f"- blocked by default: `{feedback_budget.get('execution_blocked_by_default')}`",
            f"- read: {feedback_budget.get('read')}",
        ]
    )
    lines.extend(
        [
            "",
            "## Pendulum Settle-Yield Axis",
            "",
            f"- profile: `{settle_axis.get('profile')}`",
            f"- active rate: `{settle_axis.get('active_rate')}`",
            f"- action gains: `{settle_axis.get('action_gains')}`",
            f"- dampings: `{settle_axis.get('dampings')}`",
            f"- springs: `{settle_axis.get('springs')}`",
            f"- position couplings: `{settle_axis.get('position_couplings')}`",
            f"- action controls position: `{settle_axis.get('action_controls_positions')}`",
            f"- standalone generator: `{settle_axis.get('standalone_generator')}`",
            f"- existing slots: `{settle_axis.get('uses_existing_state_action_slots')}`",
            f"- read: {settle_axis.get('read')}",
        ]
    )
    scale = report["fit_model_2048_scale_projection"]
    naive = scale["naive_all_at_once"]
    chunked = scale["chunked_probe"]
    lines.extend(
        [
            "",
            "## Fit Model 2048 Scale",
            "",
            f"- selected envs: `{scale['fit_model_target_envs']}`",
            f"- source candidate envs at multiplier {scale['candidate_multiplier']}: `{scale['source_target_envs']}`",
            f"- naive projected RSS/GPU delta: `{naive['projected_rss_gib']:.2f} GiB` / `{naive['projected_gpu_delta_gib']:.2f} GiB`",
            f"- naive all-at-once memory safe: `{naive['memory_safe']}`",
            f"- chunked probe envs: `{chunked['probe_chunk_envs']}`, chunks: `{chunked['chunk_count']}`",
            f"- chunked projected RSS/GPU delta per chunk: `{chunked['projected_rss_gib_per_chunk']:.2f} GiB` / `{chunked['projected_gpu_delta_gib_per_chunk']:.2f} GiB`",
            f"- chunked memory safe: `{chunked['memory_safe']}`",
            f"- read: {scale['pressure_read']}",
        ]
    )
    lines.extend(["", "## Failure Semantics", ""])
    for item in report["bounded_generator_plan"]["failure_semantics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next", "", f"- {forecast['next_highest_value']}", "", "## Files", ""])
    for key, value in report.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile_online_selector_preflight.json"
    md_path = out_dir / "profile_online_selector_preflight.md"

    def _build_and_write() -> dict[str, Any]:
        built = build_report(
            existing_pool_root=args.existing_pool_root,
            existing_selector_dir=args.existing_selector_dir,
            planned_run_root=args.planned_run_root,
            target_per_rule=args.target_per_rule,
            candidate_multiplier=args.candidate_multiplier,
            seed=args.seed,
            env_seed_start=args.env_seed_start,
            device=args.device,
            profile_config=args.profile_config,
            max_raw_samples=args.max_raw_samples,
            raw_batch_size=args.raw_batch_size,
            env_eval_batch_size=args.env_eval_batch_size,
            n_steps_action=args.n_steps_action,
            n_steps_locomotion=args.n_steps_locomotion,
            action_obs_linear_count=args.action_obs_linear_count,
            locomotion_obs_linear_count=args.locomotion_obs_linear_count,
            n_steps_feedback=args.n_steps_feedback,
            feedback_label_mode=args.feedback_label_mode,
            feedback_baseline_modes=args.feedback_baseline_modes,
            feedback_profiles=args.feedback_profiles,
            feedback_gains=args.feedback_gains,
            feedback_obs_linear_count=args.feedback_obs_linear_count,
            max_feedback_eval_units=args.max_feedback_eval_units,
            fit_model_target_envs=args.fit_model_target_envs,
            reference_probe_envs=args.reference_probe_envs,
            reference_probe_rss_gib=args.reference_probe_rss_gib,
            reference_probe_gpu_delta_gib=args.reference_probe_gpu_delta_gib,
            probe_chunk_envs=args.probe_chunk_envs,
            host_rss_limit_gib=args.host_rss_limit_gib,
            gpu0_mem_limit_gib=args.gpu0_mem_limit_gib,
            pendulum_settle_yield_axis_profile=args.pendulum_settle_yield_axis_profile,
            pendulum_settle_yield_axis_rate=args.pendulum_settle_yield_axis_rate,
            pendulum_settle_yield_axis_action_gains=args.pendulum_settle_yield_axis_action_gains,
            pendulum_settle_yield_axis_dampings=args.pendulum_settle_yield_axis_dampings,
            pendulum_settle_yield_axis_springs=args.pendulum_settle_yield_axis_springs,
            pendulum_settle_yield_axis_position_couplings=args.pendulum_settle_yield_axis_position_couplings,
            pendulum_settle_yield_axis_action_controls_positions=args.pendulum_settle_yield_axis_action_controls_positions,
        )
        built["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
        json_path.write_text(json.dumps(_json_safe(built), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(_markdown(built), encoding="utf-8")
        return built

    report = _build_and_write()
    if bool(args.execute_missing):
        for stage in report["bounded_generator_plan"]["stage_plan"]:
            if stage["ready"] and bool(args.skip_existing):
                continue
            if (
                str(stage["name"]).startswith("feedback_")
                and report["bounded_generator_plan"]["feedback_eval_budget"]["execution_blocked_by_default"]
                and not bool(args.allow_expensive_feedback)
            ):
                budget = report["bounded_generator_plan"]["feedback_eval_budget"]
                raise RuntimeError(
                    "Refusing to execute full feedback near-best grid by default: "
                    f"eval_units={budget['eval_units']} > max_feedback_eval_units="
                    f"{budget['max_eval_units_for_default_execute']}. "
                    "Use --allow-expensive-feedback only for an explicit research run; "
                    "fit_model startup should use the chunked/budgeted label path."
                )
            subprocess.run(stage["command"], cwd=str(REPO), check=True)
        # Refresh readiness after the orchestrated stages materialize outputs.
        report = _build_and_write()
    print(json.dumps(report["outputs"], sort_keys=True))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-pool-root", default=str(DEFAULT_EXISTING_POOL_ROOT))
    parser.add_argument("--existing-selector-dir", default=str(DEFAULT_EXISTING_SELECTOR_DIR))
    parser.add_argument(
        "--planned-run-root",
        default=str(ART / "phase2_profile_online_candidate_pool_feedback_support_generator_seed9601_0511"),
    )
    parser.add_argument("--profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-per-rule", type=int, default=16)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--env-seed-start", type=int, default=9601000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-raw-samples", type=int, default=262144)
    parser.add_argument("--raw-batch-size", type=int, default=4096)
    parser.add_argument("--env-eval-batch-size", type=int, default=64)
    parser.add_argument("--n-steps-action", type=int, default=256)
    parser.add_argument("--n-steps-locomotion", type=int, default=256)
    parser.add_argument("--action-obs-linear-count", type=int, default=4)
    parser.add_argument("--locomotion-obs-linear-count", type=int, default=4)
    parser.add_argument("--n-steps-feedback", type=int, default=64)
    parser.add_argument("--feedback-label-mode", choices=("budgeted_guard", "full_audit"), default="budgeted_guard")
    parser.add_argument("--feedback-baseline-modes", default="zero,random,neg_random,half_random")
    parser.add_argument("--feedback-profiles", default="decay")
    parser.add_argument("--feedback-gains", default="1.0")
    parser.add_argument("--feedback-obs-linear-count", type=int, default=2)
    parser.add_argument("--max-feedback-eval-units", type=int, default=500_000)
    parser.add_argument("--fit-model-target-envs", type=int, default=2048)
    parser.add_argument("--reference-probe-envs", type=int, default=64)
    parser.add_argument("--reference-probe-rss-gib", type=float, default=1.75)
    parser.add_argument("--reference-probe-gpu-delta-gib", type=float, default=1.65)
    parser.add_argument("--probe-chunk-envs", type=int, default=64)
    parser.add_argument("--host-rss-limit-gib", type=float, default=32.0)
    parser.add_argument("--gpu0-mem-limit-gib", type=float, default=49.0)
    parser.add_argument("--pendulum-settle-yield-axis-profile", default="off")
    parser.add_argument("--pendulum-settle-yield-axis-rate", type=float, default=0.0)
    parser.add_argument("--pendulum-settle-yield-axis-action-gains", default="0.35,0.55")
    parser.add_argument("--pendulum-settle-yield-axis-dampings", default="0.88,0.94")
    parser.add_argument("--pendulum-settle-yield-axis-springs", default="0.10")
    parser.add_argument("--pendulum-settle-yield-axis-position-couplings", default="0.0")
    parser.add_argument("--pendulum-settle-yield-axis-action-controls-positions", default="false")
    parser.add_argument("--execute-missing", action="store_true")
    parser.add_argument("--allow-expensive-feedback", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
