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
from ticl.analysis.phase3_env_delta_concentration_probe import (
    _compute_concentration,
    _load_reused_per_env_metrics,
)
from ticl.analysis.phase3_heldout_token_bucket_probe import _load_heldout_subset_deltas
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _default_device,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_train_rollout_quality_probe import (
    _collect_one_rollout,
    _count_objective_episodes,
    _global_token_mass_summary,
    _safe_corrcoef,
    _safe_spearman,
    _topk_abs_share,
    _window_positive_share,
)
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config, _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    _explained_variance_with_mask,
    _normalize_advantages_with_mask,
    build_recurrent_ppo,
)
from stable_baselines3.common.logger import configure as configure_logger


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 lightweight heldout-rollout quality probe for many-env transfer bottleneck analysis."
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

    num_features = int(config["prior"]["num_features"])
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )

    reused_zero = _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_zero_control_json),
        expected_policy_mode="zero",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
    )
    reused_pre = _load_reused_per_env_metrics(
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
        num_features=num_features,
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
    raw_values = (
        rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )
    raw_returns = (
        rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )

    objective_flat = torch.as_tensor(objective_masks.reshape(-1) > 1e-8, dtype=torch.bool)
    actor_adv_flat = torch.as_tensor(actor_advantages.reshape(-1), dtype=torch.float32)
    normalized_adv_flat = (
        _normalize_advantages_with_mask(actor_adv_flat, objective_flat, eps=1e-8)
        if bool(getattr(algo, "normalize_advantage", False))
        else actor_adv_flat
    )
    normalized_actor_advantages = normalized_adv_flat.reshape(objective_masks.shape).detach().cpu().numpy()
    vec_env.close()

    env_items: list[dict[str, Any]] = []
    env_positive_norm_mass = []
    env_abs_norm_mass = []
    env_pre_suffix_gap = []
    env_value_corr = []
    env_terminal_resets = []
    env_last16_share = []
    subset_metric_rows: list[dict[str, Any]] = []

    zero_full = reused_zero["heldout"]["full_return_per_env"]
    zero_suffix = reused_zero["heldout"]["suffix_return_per_env"]
    pre_full = reused_pre["heldout"]["full_return_per_env"]
    pre_suffix = reused_pre["heldout"]["suffix_return_per_env"]

    for env_idx in range(int(objective_masks.shape[1])):
        obj = objective_masks[:, env_idx] > 1e-8
        obj_adv = actor_advantages[:, env_idx][obj]
        obj_norm_adv = normalized_actor_advantages[:, env_idx][obj]
        obj_values = raw_values[:, env_idx][obj]
        obj_returns = raw_returns[:, env_idx][obj]
        episode_count, terminal_resets = _count_objective_episodes(obj, episode_starts[:, env_idx] > 1e-8)
        positive_norm_mass = float(np.clip(obj_norm_adv, a_min=0.0, a_max=None).sum())
        negative_norm_mass = float(np.clip(-obj_norm_adv, a_min=0.0, a_max=None).sum())
        abs_norm_mass = float(np.abs(obj_norm_adv).sum())
        positive_raw_mass = float(np.clip(obj_adv, a_min=0.0, a_max=None).sum())
        negative_raw_mass = float(np.clip(-obj_adv, a_min=0.0, a_max=None).sum())
        abs_raw_mass = float(np.abs(obj_adv).sum())
        value_corr = _safe_corrcoef(obj_values.tolist(), obj_returns.tolist()) if int(obj.sum()) > 1 else 0.0
        value_ev = (
            float(
                _explained_variance_with_mask(
                    obj_values,
                    obj_returns,
                    mask=np.ones_like(obj_values, dtype=np.float32),
                )
            )
            if int(obj.sum()) > 1
            else 0.0
        )
        item = {
            "env_index": int(env_idx),
            "env_seed": int(list(heldout_suite["env_seeds"])[env_idx]),
            "rollout_seed": int(list(heldout_suite["rollout_seeds"])[env_idx]),
            "objective_token_count": int(obj.sum()),
            "objective_token_fraction": float(obj.mean()),
            "objective_episode_count": int(episode_count),
            "objective_terminal_reset_count": int(terminal_resets),
            "pre_full_return": float(pre_full[env_idx]),
            "zero_full_return": float(zero_full[env_idx]),
            "pre_suffix_return": float(pre_suffix[env_idx]),
            "zero_suffix_return": float(zero_suffix[env_idx]),
            "pre_vs_zero_full_gap": float(pre_full[env_idx] - zero_full[env_idx]),
            "pre_vs_zero_suffix_gap": float(pre_suffix[env_idx] - zero_suffix[env_idx]),
            "raw_value_return_corr": float(value_corr),
            "raw_value_explained_variance": float(value_ev),
            "raw_objective_return_mean": float(obj_returns.mean()) if int(obj.sum()) > 0 else 0.0,
            "raw_objective_return_std": float(obj_returns.std()) if int(obj.sum()) > 0 else 0.0,
            "raw_actor_adv_positive_mass": positive_raw_mass,
            "raw_actor_adv_negative_mass": negative_raw_mass,
            "raw_actor_adv_abs_mass": abs_raw_mass,
            "normalized_actor_adv_positive_mass": positive_norm_mass,
            "normalized_actor_adv_negative_mass": negative_norm_mass,
            "normalized_actor_adv_abs_mass": abs_norm_mass,
            "normalized_actor_adv_positive_count": int(np.count_nonzero(obj_norm_adv > 0.0)),
            "normalized_actor_adv_negative_count": int(np.count_nonzero(obj_norm_adv < 0.0)),
            "normalized_actor_adv_top1_abs_share": _topk_abs_share(obj_norm_adv, 1),
            "normalized_actor_adv_top4_abs_share": _topk_abs_share(obj_norm_adv, 4),
            "normalized_actor_adv_top16_abs_share": _topk_abs_share(obj_norm_adv, 16),
            "normalized_actor_adv_last16_positive_share": _window_positive_share(
                obj_norm_adv,
                take_last=True,
                width=16,
            ),
            "normalized_actor_adv_first16_positive_share": _window_positive_share(
                obj_norm_adv,
                take_last=False,
                width=16,
            ),
        }
        if env_idx in subset_deltas:
            item.update(subset_deltas[env_idx])
            subset_metric_rows.append(item)
        env_items.append(item)
        env_positive_norm_mass.append(positive_norm_mass)
        env_abs_norm_mass.append(abs_norm_mass)
        env_pre_suffix_gap.append(float(item["pre_vs_zero_suffix_gap"]))
        env_value_corr.append(float(item["raw_value_return_corr"]))
        env_terminal_resets.append(float(item["objective_terminal_reset_count"]))
        env_last16_share.append(float(item["normalized_actor_adv_last16_positive_share"]))

    positive_mass_ranking = sorted(
        env_items,
        key=lambda item: float(item["normalized_actor_adv_positive_mass"]),
        reverse=True,
    )
    subset_alignment = {}
    if subset_metric_rows:
        subset_suffix_gap_delta = [float(item["suffix_gap_delta"]) for item in subset_metric_rows]
        subset_alignment = {
            "subset_env_indices": [int(item["env_index"]) for item in subset_metric_rows],
            "corr_suffix_gap_delta_vs_pre_suffix_gap": _safe_corrcoef(
                [float(item["pre_vs_zero_suffix_gap"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "spearman_suffix_gap_delta_vs_pre_suffix_gap": _safe_spearman(
                [float(item["pre_vs_zero_suffix_gap"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "corr_suffix_gap_delta_vs_positive_norm_mass": _safe_corrcoef(
                [float(item["normalized_actor_adv_positive_mass"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "spearman_suffix_gap_delta_vs_positive_norm_mass": _safe_spearman(
                [float(item["normalized_actor_adv_positive_mass"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "corr_suffix_gap_delta_vs_value_corr": _safe_corrcoef(
                [float(item["raw_value_return_corr"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "corr_suffix_gap_delta_vs_terminal_resets": _safe_corrcoef(
                [float(item["objective_terminal_reset_count"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
            "corr_suffix_gap_delta_vs_last16_positive_share": _safe_corrcoef(
                [float(item["normalized_actor_adv_last16_positive_share"]) for item in subset_metric_rows],
                subset_suffix_gap_delta,
            ),
        }

    objective_count_uniform = len({int(item["objective_token_count"]) for item in env_items}) == 1
    positive_mass_concentration = _compute_concentration(env_positive_norm_mass)

    report = {
        "audit_entry": "phase3_heldout_rollout_quality_probe",
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
            "train_profile": str(profile_cfg["train_profile"]),
            "profile_build_n_envs": int(heldout_suite["batch_size"]),
            "profile_build_n_steps": int(args.n_samples),
            "profile_build_batch_size": int(profile_cfg["batch_size"]),
        },
        "runtime_wall_s": float(time.perf_counter() - t0),
        "train_suite_summary": train_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "rollout_summary": {
            "objective_token_total": int(np.count_nonzero(objective_masks > 1e-8)),
            "per_env_objective_token_count_stats": _compute_concentration(
                [float(item["objective_token_count"]) for item in env_items]
            ),
            "per_env_terminal_reset_count_stats": _compute_concentration(env_terminal_resets),
            "per_env_pre_suffix_gap_stats": _compute_concentration(env_pre_suffix_gap),
            "per_env_positive_norm_mass_stats": _compute_concentration(env_positive_norm_mass),
            "per_env_abs_norm_mass_stats": _compute_concentration(env_abs_norm_mass),
            "corr_pre_suffix_gap_vs_positive_norm_mass": _safe_corrcoef(env_pre_suffix_gap, env_positive_norm_mass),
            "corr_value_corr_vs_positive_norm_mass": _safe_corrcoef(env_value_corr, env_positive_norm_mass),
            "corr_terminal_resets_vs_positive_norm_mass": _safe_corrcoef(env_terminal_resets, env_positive_norm_mass),
            "corr_last16_positive_share_vs_positive_norm_mass": _safe_corrcoef(
                env_last16_share,
                env_positive_norm_mass,
            ),
            "global_token_positive_mass": _global_token_mass_summary(
                normalized_actor_advantages=normalized_actor_advantages,
                objective_masks=objective_masks > 1e-8,
            ),
            "top_envs_by_positive_norm_mass": [
                {
                    "env_index": int(item["env_index"]),
                    "pre_vs_zero_suffix_gap": float(item["pre_vs_zero_suffix_gap"]),
                    "normalized_actor_adv_positive_mass": float(item["normalized_actor_adv_positive_mass"]),
                    "normalized_actor_adv_last16_positive_share": float(
                        item["normalized_actor_adv_last16_positive_share"]
                    ),
                    "objective_terminal_reset_count": int(item["objective_terminal_reset_count"]),
                    "raw_value_return_corr": float(item["raw_value_return_corr"]),
                }
                for item in positive_mass_ranking[:4]
            ],
        },
        "env_items": env_items,
        "subset_alignment": subset_alignment,
        "conclusions": {
            "phase2_guardrail_passed": True,
            "objective_token_count_uniform": bool(objective_count_uniform),
            "positive_mass_is_concentrated": bool(
                positive_mass_concentration["top1_abs_mass_share"] > 0.20
            ),
            "subset_positive_mass_not_main_driver": bool(
                subset_alignment.get("corr_suffix_gap_delta_vs_positive_norm_mass", 0.0) < 0.5
            ),
            "subset_value_corr_more_aligned_than_positive_mass": bool(
                subset_alignment.get("corr_suffix_gap_delta_vs_value_corr", 0.0)
                > subset_alignment.get("corr_suffix_gap_delta_vs_positive_norm_mass", 0.0)
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
