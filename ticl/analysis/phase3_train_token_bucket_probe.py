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
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_train_rollout_quality_probe import (
    _collect_one_rollout,
    _load_subset_deltas,
    _safe_corrcoef,
)
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config, _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


BUCKET_NAMES = ("q1", "q2", "q3", "q4")


def _bucket_slices(n: int) -> list[tuple[int, int]]:
    edges = [0]
    for idx in range(1, 5):
        edges.append(int(round(idx * n / 4.0)))
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        out.append((int(start), int(max(start, end))))
    return out


def _bucket_metrics(
    *,
    normalized_adv: np.ndarray,
    raw_values: np.ndarray,
    raw_returns: np.ndarray,
) -> dict[str, dict[str, float]]:
    normalized_adv = np.asarray(normalized_adv, dtype=np.float32).reshape(-1)
    raw_values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    raw_returns = np.asarray(raw_returns, dtype=np.float32).reshape(-1)
    total_pos = float(np.clip(normalized_adv, a_min=0.0, a_max=None).sum())
    total_neg = float(np.clip(-normalized_adv, a_min=0.0, a_max=None).sum())
    bucket_stats: dict[str, dict[str, float]] = {}
    for bucket_name, (start, end) in zip(BUCKET_NAMES, _bucket_slices(int(normalized_adv.size))):
        bucket_adv = normalized_adv[start:end]
        bucket_values = raw_values[start:end]
        bucket_returns = raw_returns[start:end]
        pos_mass = float(np.clip(bucket_adv, a_min=0.0, a_max=None).sum())
        neg_mass = float(np.clip(-bucket_adv, a_min=0.0, a_max=None).sum())
        bucket_stats[bucket_name] = {
            "token_count": float(int(end - start)),
            "positive_mass": pos_mass,
            "negative_mass": neg_mass,
            "positive_mass_share": float(pos_mass / total_pos) if total_pos > 0.0 else 0.0,
            "negative_mass_share": float(neg_mass / total_neg) if total_neg > 0.0 else 0.0,
            "signed_mean": float(bucket_adv.mean()) if int(bucket_adv.size) > 0 else 0.0,
            "raw_value_return_corr": (
                _safe_corrcoef(bucket_values.tolist(), bucket_returns.tolist())
                if int(bucket_adv.size) > 1
                else 0.0
            ),
        }
    return bucket_stats


def _mean(rows: list[dict[str, Any]], getter) -> float:
    if not rows:
        return 0.0
    return float(sum(float(getter(row)) for row in rows) / float(len(rows)))


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 lightweight objective token-bucket probe on the train-suite rollout."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--reuse-zero-control-json", type=str, required=True)
    parser.add_argument("--reuse-pre-policy-json", type=str, required=True)
    parser.add_argument("--reuse-train-subset-delta-json", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
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
    subset_deltas = _load_subset_deltas(str(args.reuse_train_subset_delta_json))

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: [copy.deepcopy(h) for h in list(train_suite["h_list"])]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args.single_eval_pos)
    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(train_suite["batch_size"]),
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
    _bind_vec_env_to_fixed_suite(vec_env, suite=train_suite, single_eval_pos=int(args.single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    _collect_one_rollout(algo, callback, vec_env)

    rollout_buffer = algo.rollout_buffer
    objective_masks = np.asarray(rollout_buffer.objective_masks, dtype=np.float32)
    actor_advantages = np.asarray(rollout_buffer.actor_advantages, dtype=np.float32)
    raw_values = rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).cpu().numpy().reshape(objective_masks.shape)
    raw_returns = rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy().reshape(objective_masks.shape)
    vec_env.close()

    subset_rows = []
    for env_idx in sorted(subset_deltas.keys()):
        obj = objective_masks[:, env_idx] > 1e-8
        env_buckets = _bucket_metrics(
            normalized_adv=actor_advantages[:, env_idx][obj],
            raw_values=raw_values[:, env_idx][obj],
            raw_returns=raw_returns[:, env_idx][obj],
        )
        row = {
            "env_index": int(env_idx),
            **subset_deltas[env_idx],
            "bucket_metrics": env_buckets,
        }
        subset_rows.append(row)

    subset_deltas_list = [float(row["suffix_gap_delta"]) for row in subset_rows]
    bucket_alignment = {}
    for bucket_name in BUCKET_NAMES:
        bucket_alignment[bucket_name] = {
            "corr_delta_vs_positive_mass_share": _safe_corrcoef(
                [float(row["bucket_metrics"][bucket_name]["positive_mass_share"]) for row in subset_rows],
                subset_deltas_list,
            ),
            "corr_delta_vs_negative_mass_share": _safe_corrcoef(
                [float(row["bucket_metrics"][bucket_name]["negative_mass_share"]) for row in subset_rows],
                subset_deltas_list,
            ),
            "corr_delta_vs_value_corr": _safe_corrcoef(
                [float(row["bucket_metrics"][bucket_name]["raw_value_return_corr"]) for row in subset_rows],
                subset_deltas_list,
            ),
        }

    # Use the already locked grouping from the subset itself: positive mass median over the subset rows.
    subset_mass = [
        float(
            sum(float(bucket["positive_mass"]) for bucket in row["bucket_metrics"].values())
        )
        for row in subset_rows
    ]
    subset_mass_median = float(np.median(np.asarray(subset_mass, dtype=np.float32)))
    high_mass_nonpositive = [
        row
        for row, mass in zip(subset_rows, subset_mass)
        if mass > subset_mass_median and float(row["suffix_gap_delta"]) <= 0.0
    ]
    low_mass_positive = [
        row
        for row, mass in zip(subset_rows, subset_mass)
        if mass <= subset_mass_median and float(row["suffix_gap_delta"]) > 0.0
    ]

    contrast = {}
    for bucket_name in BUCKET_NAMES:
        contrast[bucket_name] = {
            "positive_mass_share": _mean(low_mass_positive, lambda row: row["bucket_metrics"][bucket_name]["positive_mass_share"])
            - _mean(high_mass_nonpositive, lambda row: row["bucket_metrics"][bucket_name]["positive_mass_share"]),
            "negative_mass_share": _mean(low_mass_positive, lambda row: row["bucket_metrics"][bucket_name]["negative_mass_share"])
            - _mean(high_mass_nonpositive, lambda row: row["bucket_metrics"][bucket_name]["negative_mass_share"]),
            "value_corr": _mean(low_mass_positive, lambda row: row["bucket_metrics"][bucket_name]["raw_value_return_corr"])
            - _mean(high_mass_nonpositive, lambda row: row["bucket_metrics"][bucket_name]["raw_value_return_corr"]),
        }

    report = {
        "audit_entry": "phase3_train_token_bucket_probe",
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
            "reuse_train_subset_delta_json": str(Path(args.reuse_train_subset_delta_json).expanduser().resolve()),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "train_profile": str(profile_cfg["train_profile"]),
            "runtime_wall_s": float(time.perf_counter() - t0),
        },
        "subset_rows": subset_rows,
        "bucket_alignment": bucket_alignment,
        "bucket_contrast_low_mass_positive_minus_high_mass_nonpositive": contrast,
        "conclusions": {
            "new_phase3_trusted_baseline_established": False,
            "subset_only_diagnostic": True,
            "q1_positive_mass_share_tracks_delta_best": bool(
                float(bucket_alignment["q1"]["corr_delta_vs_positive_mass_share"])
                > float(bucket_alignment["q4"]["corr_delta_vs_positive_mass_share"])
            ),
            "q4_tail_positive_mass_share_not_primary_driver": bool(
                abs(float(bucket_alignment["q4"]["corr_delta_vs_positive_mass_share"])) < 0.3
            ),
            "bucketed_value_corr_supports_early_buckets_more_than_tail": bool(
                float(contrast["q1"]["value_corr"]) > float(contrast["q4"]["value_corr"])
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
