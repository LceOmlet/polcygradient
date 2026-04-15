import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase3_env_delta_concentration_probe import _load_reused_per_env_metrics
from ticl.analysis.phase3_heldout_token_bucket_probe import _load_heldout_subset_deltas
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_train_rollout_quality_probe import _collect_one_rollout
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config, _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


SEGMENT_NAMES = (
    "first_episode",
    "post_first_reset",
    "terminal_tail",
    "non_tail",
    "first_episode_terminal_tail",
    "first_episode_non_tail",
    "post_reset_terminal_tail",
    "post_reset_non_tail",
)


def _objective_episode_segments(objective_mask: np.ndarray, episode_starts: np.ndarray) -> list[tuple[int, int]]:
    objective_mask = np.asarray(objective_mask, dtype=bool).reshape(-1)
    episode_starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
    valid_len = int(objective_mask.sum())
    if valid_len <= 0:
        return []
    valid_starts = episode_starts[objective_mask]
    start_positions = np.flatnonzero(valid_starts).tolist()
    if not start_positions or int(start_positions[0]) != 0:
        start_positions = [0] + start_positions
    starts = sorted({int(v) for v in start_positions if 0 <= int(v) < valid_len})
    segments: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else valid_len
        if int(end) > int(start):
            segments.append((int(start), int(end)))
    return segments


def _objective_terminal_reset_count_from_segments(segments: list[tuple[int, int]]) -> int:
    # Segment count is defined on the compressed objective-token timeline, so reset count
    # must be derived from those segments rather than raw episode_starts in the full rollout.
    return max(0, int(len(segments) - 1))


def _segment_masks(valid_len: int, segments: list[tuple[int, int]], tail_len: int) -> dict[str, np.ndarray]:
    first_episode = np.zeros((valid_len,), dtype=bool)
    post_first_reset = np.zeros((valid_len,), dtype=bool)
    terminal_tail = np.zeros((valid_len,), dtype=bool)
    if segments:
        s0, e0 = segments[0]
        first_episode[s0:e0] = True
    if len(segments) > 1:
        s1, _ = segments[1]
        post_first_reset[s1:] = True
    for start, end in segments:
        tail_start = max(int(start), int(end) - int(tail_len))
        terminal_tail[tail_start:end] = True
    non_tail = ~terminal_tail
    return {
        "first_episode": first_episode,
        "post_first_reset": post_first_reset,
        "terminal_tail": terminal_tail,
        "non_tail": non_tail,
        "first_episode_terminal_tail": first_episode & terminal_tail,
        "first_episode_non_tail": first_episode & non_tail,
        "post_reset_terminal_tail": post_first_reset & terminal_tail,
        "post_reset_non_tail": post_first_reset & non_tail,
    }


def _segment_summary(
    *,
    actor_adv: np.ndarray,
    raw_returns: np.ndarray,
    raw_values: np.ndarray,
    mask: np.ndarray,
    total_pos_actor_adv_mass: float,
    total_pos_residual_mass: float,
) -> dict[str, float]:
    actor_adv = np.asarray(actor_adv, dtype=np.float32).reshape(-1)
    raw_returns = np.asarray(raw_returns, dtype=np.float32).reshape(-1)
    raw_values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.sum()) <= 0:
        return {
            "token_count": 0.0,
            "actor_adv_positive_mass": 0.0,
            "actor_adv_positive_mass_share": 0.0,
            "actor_adv_abs_mass": 0.0,
            "actor_adv_positive_count": 0.0,
            "raw_return_mean": 0.0,
            "raw_value_mean": 0.0,
            "raw_residual_mean": 0.0,
            "raw_residual_positive_mass": 0.0,
            "raw_residual_positive_mass_share": 0.0,
        }
    seg_adv = actor_adv[mask]
    seg_returns = raw_returns[mask]
    seg_values = raw_values[mask]
    residual = seg_returns - seg_values
    pos_adv = float(np.clip(seg_adv, a_min=0.0, a_max=None).sum())
    pos_residual = float(np.clip(residual, a_min=0.0, a_max=None).sum())
    return {
        "token_count": float(int(mask.sum())),
        "actor_adv_positive_mass": pos_adv,
        "actor_adv_positive_mass_share": float(pos_adv / total_pos_actor_adv_mass) if total_pos_actor_adv_mass > 0.0 else 0.0,
        "actor_adv_abs_mass": float(np.abs(seg_adv).sum()),
        "actor_adv_positive_count": float(int(np.count_nonzero(seg_adv > 0.0))),
        "raw_return_mean": float(seg_returns.mean()),
        "raw_value_mean": float(seg_values.mean()),
        "raw_residual_mean": float(residual.mean()),
        "raw_residual_positive_mass": pos_residual,
        "raw_residual_positive_mass_share": float(pos_residual / total_pos_residual_mass)
        if total_pos_residual_mass > 0.0
        else 0.0,
    }


def _aggregate_segment_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {name: {} for name in SEGMENT_NAMES}
    out: dict[str, dict[str, float]] = {}
    for name in SEGMENT_NAMES:
        sample = rows[0]["segment_metrics"][name]
        out[name] = {
            key: float(sum(float(row["segment_metrics"][name][key]) for row in rows) / len(rows))
            for key in sample.keys()
        }
    return out


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 heldout reset-semantics probe under the current lightweight many-env contract."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--reuse-zero-control-json", type=str, required=True)
    parser.add_argument("--reuse-pre-policy-json", type=str, required=True)
    parser.add_argument("--reuse-heldout-subset-delta-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--tail-len", type=int, default=8)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    phase2_summary = assert_phase2_green(args.phase2_summary_path)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(args.device or _default_device()))

    train_suite = _load_required_suite(args.train_suite_path)
    heldout_suite = _load_required_suite(args.heldout_suite_path)

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    eval_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    eval_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args.single_eval_pos)
    train_suite_summary = _summarize_suite(eval_prior, train_suite)
    heldout_suite_summary = _summarize_suite(eval_prior, heldout_suite)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )

    _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_zero_control_json),
        expected_policy_mode="zero",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
    )
    _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_pre_policy_json),
        expected_policy_mode="ppo",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
    )
    subset_deltas = (
        {}
        if args.reuse_heldout_subset_delta_json is None
        else _load_heldout_subset_deltas(str(args.reuse_heldout_subset_delta_json))
    )

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(heldout_suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args.single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(heldout_suite["batch_size"]),
        n_steps=int(args.n_samples),
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=int(profile_cfg["batch_size"]),
        n_epochs=int(profile_cfg["n_epochs"]),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(profile_cfg["normalize_advantage"]),
        actor_gae_space=str(profile_cfg["actor_gae_space"]),
        actor_baseline_mode=str(profile_cfg["actor_baseline_mode"]),
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=heldout_suite, single_eval_pos=int(args.single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _collect_one_rollout(algo, callback, vec_env)

    rollout_buffer = algo.rollout_buffer
    objective_masks = np.asarray(rollout_buffer.objective_masks, dtype=np.float32)
    episode_starts = np.asarray(rollout_buffer.episode_starts, dtype=np.float32)
    actor_advantages = np.asarray(rollout_buffer.actor_advantages, dtype=np.float32)
    raw_values = rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).cpu().numpy().reshape(objective_masks.shape)
    raw_returns = rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy().reshape(objective_masks.shape)
    vec_env.close()

    env_rows = []
    for env_idx in range(int(objective_masks.shape[1])):
        objective_mask = objective_masks[:, env_idx] > 1e-8
        if int(objective_mask.sum()) <= 0:
            continue
        env_episode_starts = episode_starts[:, env_idx] > 1e-8
        env_adv = actor_advantages[:, env_idx][objective_mask]
        env_returns = raw_returns[:, env_idx][objective_mask]
        env_values = raw_values[:, env_idx][objective_mask]
        segments = _objective_episode_segments(objective_mask, env_episode_starts)
        masks = _segment_masks(int(objective_mask.sum()), segments, int(args.tail_len))
        total_pos_adv = float(np.clip(env_adv, a_min=0.0, a_max=None).sum())
        total_pos_residual = float(np.clip(env_returns - env_values, a_min=0.0, a_max=None).sum())
        row = {
            "env_index": int(env_idx),
            "env_seed": int(list(heldout_suite["env_seeds"])[env_idx]),
            "rollout_seed": int(list(heldout_suite["rollout_seeds"])[env_idx]),
            "objective_token_count": int(objective_mask.sum()),
            "objective_terminal_reset_count": max(0, int(len(segments) - 1)),
            "pre_vs_zero_suffix_gap": float(subset_deltas.get(env_idx, {}).get("pre_vs_zero_suffix_gap", 0.0)),
            "suffix_gap_delta": float(subset_deltas.get(env_idx, {}).get("suffix_gap_delta", 0.0)),
            "raw_value_return_corr": (
                float(np.corrcoef(env_values, env_returns)[0, 1])
                if int(env_values.size) > 1 and float(np.std(env_values)) > 0.0 and float(np.std(env_returns)) > 0.0
                else 0.0
            ),
            "actor_adv_positive_mass": total_pos_adv,
            "raw_residual_positive_mass": total_pos_residual,
            "segment_metrics": {
                name: _segment_summary(
                    actor_adv=env_adv,
                    raw_returns=env_returns,
                    raw_values=env_values,
                    mask=mask,
                    total_pos_actor_adv_mass=total_pos_adv,
                    total_pos_residual_mass=total_pos_residual,
                )
                for name, mask in masks.items()
            },
        }
        env_rows.append(row)

    reset_heavy_rows = [row for row in env_rows if int(row["objective_terminal_reset_count"]) > 0]
    reset_heavy_rows = sorted(reset_heavy_rows, key=lambda row: float(row["actor_adv_positive_mass"]), reverse=True)
    top_reset_heavy = reset_heavy_rows[: min(8, len(reset_heavy_rows))]

    aggregate_means = _aggregate_segment_means(reset_heavy_rows)
    top_reset_heavy_means = _aggregate_segment_means(top_reset_heavy)

    report = {
        "audit_entry": "phase3_heldout_reset_semantics_probe",
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": str(Path(args.train_suite_path).expanduser().resolve()),
            "heldout_suite_path": str(Path(args.heldout_suite_path).expanduser().resolve()),
            "reuse_zero_control_json": str(Path(args.reuse_zero_control_json).expanduser().resolve()),
            "reuse_pre_policy_json": str(Path(args.reuse_pre_policy_json).expanduser().resolve()),
            "reuse_heldout_subset_delta_json": None
            if args.reuse_heldout_subset_delta_json is None
            else str(Path(args.reuse_heldout_subset_delta_json).expanduser().resolve()),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "tail_len": int(args.tail_len),
            "train_profile": str(profile_cfg["train_profile"]),
            "runtime_wall_s": float(time.perf_counter() - t0),
        },
        "train_suite_summary": train_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "reset_heavy_env_indices": [int(row["env_index"]) for row in reset_heavy_rows],
        "top_reset_heavy_env_rows": top_reset_heavy,
        "aggregate_segment_means_reset_heavy": aggregate_means,
        "aggregate_segment_means_top_reset_heavy": top_reset_heavy_means,
        "conclusions": {
            "post_first_reset_carries_more_positive_mass_than_first_episode": bool(
                aggregate_means["post_first_reset"].get("actor_adv_positive_mass_share", 0.0)
                > aggregate_means["first_episode"].get("actor_adv_positive_mass_share", 0.0)
            ),
            "terminal_tail_carries_more_positive_mass_than_non_tail": bool(
                aggregate_means["terminal_tail"].get("actor_adv_positive_mass_share", 0.0)
                > aggregate_means["non_tail"].get("actor_adv_positive_mass_share", 0.0)
            ),
            "post_reset_terminal_tail_is_dominant_positive_mass_sink": bool(
                aggregate_means["post_reset_terminal_tail"].get("actor_adv_positive_mass_share", 0.0)
                > max(
                    aggregate_means["post_reset_non_tail"].get("actor_adv_positive_mass_share", 0.0),
                    aggregate_means["first_episode_terminal_tail"].get("actor_adv_positive_mass_share", 0.0),
                    aggregate_means["first_episode_non_tail"].get("actor_adv_positive_mass_share", 0.0),
                )
            ),
            "new_phase3_trusted_baseline_established": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
