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
from ticl.analysis.phase3_heldout_reset_semantics_probe import (
    _objective_episode_segments,
    _objective_terminal_reset_count_from_segments,
    _segment_masks,
)
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


ABLATION_TARGETS = (
    "first_episode_non_tail",
    "post_reset_non_tail",
    "first_episode_terminal_tail",
    "post_reset_terminal_tail",
    "terminal_tail",
    "non_tail",
)


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have matching lengths.")
    if len(xs) <= 1:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _topk_mass_share(items: list[dict[str, Any]], *, mass_key: str, rank_key: str, k: int) -> dict[str, Any]:
    ranked = sorted(items, key=lambda item: float(item[rank_key]), reverse=True)
    top = ranked[: max(1, min(int(k), len(ranked)))]
    total_mass = float(sum(float(item[mass_key]) for item in items))
    top_mass = float(sum(float(item[mass_key]) for item in top))
    return {
        "k": int(k),
        "rank_key": str(rank_key),
        "mass_key": str(mass_key),
        "env_indices": [int(item["env_index"]) for item in top],
        "mean_rank_value": float(sum(float(item[rank_key]) for item in top) / max(1, len(top))),
        "mass_share": float(top_mass / total_mass) if total_mass > 0.0 else 0.0,
    }


def _summary(items: list[dict[str, Any]], *, mass_key: str) -> dict[str, Any]:
    masses = [float(item[mass_key]) for item in items]
    terminal_resets = [float(item["objective_terminal_reset_count"]) for item in items]
    pre_gap = [float(item["pre_vs_zero_suffix_gap"]) for item in items]
    value_corr = [float(item["raw_value_return_corr"]) for item in items]
    return {
        "corr_vs_terminal_resets": _corr(masses, terminal_resets),
        "corr_vs_pre_suffix_gap": _corr(masses, pre_gap),
        "corr_vs_value_corr": _corr(masses, value_corr),
        "top2_pre_gap_capture": _topk_mass_share(items, mass_key=mass_key, rank_key="pre_vs_zero_suffix_gap", k=2),
        "top4_pre_gap_capture": _topk_mass_share(items, mass_key=mass_key, rank_key="pre_vs_zero_suffix_gap", k=4),
        "mass_total": float(sum(masses)),
    }


def _ablate_mass(
    *,
    env_adv: np.ndarray,
    env_returns: np.ndarray,
    env_values: np.ndarray,
    objective_mask: np.ndarray,
    episode_starts: np.ndarray,
    tail_len: int,
    ablate_key: str,
) -> dict[str, float]:
    if int(objective_mask.sum()) <= 0:
        return {
            "token_count": 0.0,
            "raw_actor_adv_positive_mass": 0.0,
            "raw_residual_positive_mass": 0.0,
            "raw_return_positive_mass": 0.0,
        }
    segments = _objective_episode_segments(objective_mask, episode_starts)
    masks = _segment_masks(int(objective_mask.sum()), segments, int(tail_len))
    if ablate_key not in masks:
        raise KeyError(f"Unknown ablation key: {ablate_key}")
    keep_mask = ~masks[ablate_key]
    if int(keep_mask.sum()) <= 0:
        return {
            "token_count": 0.0,
            "raw_actor_adv_positive_mass": 0.0,
            "raw_residual_positive_mass": 0.0,
            "raw_return_positive_mass": 0.0,
        }
    adv = env_adv[keep_mask]
    returns = env_returns[keep_mask]
    values = env_values[keep_mask]
    residual = returns - values
    return {
        "token_count": float(int(keep_mask.sum())),
        "raw_actor_adv_positive_mass": float(np.clip(adv, a_min=0.0, a_max=None).sum()),
        "raw_residual_positive_mass": float(np.clip(residual, a_min=0.0, a_max=None).sum()),
        "raw_return_positive_mass": float(np.clip(returns, a_min=0.0, a_max=None).sum()),
    }


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 heldout segment counterfactual ablation under the lightweight many-env contract."
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
    raw_values = (
        rollout_buffer._get_flat_raw_values_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )
    raw_returns = (
        rollout_buffer._get_flat_raw_returns_tensor(dtype=torch.float32).detach().cpu().numpy().reshape(objective_masks.shape)
    )
    vec_env.close()

    env_rows: list[dict[str, Any]] = []
    for env_idx in range(int(objective_masks.shape[1])):
        obj = objective_masks[:, env_idx] > 1e-8
        if int(obj.sum()) <= 0:
            continue
        env_adv = actor_advantages[:, env_idx][obj]
        env_returns = raw_returns[:, env_idx][obj]
        env_values = raw_values[:, env_idx][obj]
        env_episode_starts = episode_starts[:, env_idx] > 1e-8
        row = {
            "env_index": int(env_idx),
            "env_seed": int(list(heldout_suite["env_seeds"])[env_idx]),
            "rollout_seed": int(list(heldout_suite["rollout_seeds"])[env_idx]),
            "objective_token_count": int(obj.sum()),
            "objective_terminal_reset_count": int(_objective_terminal_reset_count_from_segments(segments)),
            "pre_vs_zero_suffix_gap": float(subset_deltas.get(env_idx, {}).get("pre_vs_zero_suffix_gap", 0.0)),
            "suffix_gap_delta": float(subset_deltas.get(env_idx, {}).get("suffix_gap_delta", 0.0)),
            "raw_value_return_corr": (
                float(np.corrcoef(env_values, env_returns)[0, 1])
                if int(env_values.size) > 1 and float(np.std(env_values)) > 0.0 and float(np.std(env_returns)) > 0.0
                else 0.0
            ),
        }
        base_mass = _ablate_mass(
            env_adv=env_adv,
            env_returns=env_returns,
            env_values=env_values,
            objective_mask=obj,
            episode_starts=env_episode_starts,
            tail_len=int(args.tail_len),
            ablate_key="terminal_tail",
        )
        # base_mass above is a convenience; override with full objective (no ablation).
        base_mass["token_count"] = float(int(obj.sum()))
        base_mass["raw_actor_adv_positive_mass"] = float(np.clip(env_adv, a_min=0.0, a_max=None).sum())
        base_mass["raw_residual_positive_mass"] = float(np.clip(env_returns - env_values, a_min=0.0, a_max=None).sum())
        base_mass["raw_return_positive_mass"] = float(np.clip(env_returns, a_min=0.0, a_max=None).sum())
        row["base_mass"] = base_mass
        row["ablations"] = {
            key: _ablate_mass(
                env_adv=env_adv,
                env_returns=env_returns,
                env_values=env_values,
                objective_mask=obj,
                episode_starts=env_episode_starts,
                tail_len=int(args.tail_len),
                ablate_key=key,
            )
            for key in ABLATION_TARGETS
        }
        env_rows.append(row)

    base_items = [
        {
            **{k: row[k] for k in ("env_index", "objective_terminal_reset_count", "pre_vs_zero_suffix_gap", "raw_value_return_corr")},
            "raw_actor_adv_positive_mass": row["base_mass"]["raw_actor_adv_positive_mass"],
            "raw_residual_positive_mass": row["base_mass"]["raw_residual_positive_mass"],
            "raw_return_positive_mass": row["base_mass"]["raw_return_positive_mass"],
        }
        for row in env_rows
    ]
    base_summary = {
        "raw_actor_adv_positive_mass": _summary(base_items, mass_key="raw_actor_adv_positive_mass"),
        "raw_residual_positive_mass": _summary(base_items, mass_key="raw_residual_positive_mass"),
        "raw_return_positive_mass": _summary(base_items, mass_key="raw_return_positive_mass"),
    }

    ablation_summaries: dict[str, dict[str, Any]] = {}
    for key in ABLATION_TARGETS:
        items = [
            {
                **{k: row[k] for k in ("env_index", "objective_terminal_reset_count", "pre_vs_zero_suffix_gap", "raw_value_return_corr")},
                "raw_actor_adv_positive_mass": row["ablations"][key]["raw_actor_adv_positive_mass"],
                "raw_residual_positive_mass": row["ablations"][key]["raw_residual_positive_mass"],
                "raw_return_positive_mass": row["ablations"][key]["raw_return_positive_mass"],
            }
            for row in env_rows
        ]
        ablation_summaries[key] = {
            "raw_actor_adv_positive_mass": _summary(items, mass_key="raw_actor_adv_positive_mass"),
            "raw_residual_positive_mass": _summary(items, mass_key="raw_residual_positive_mass"),
            "raw_return_positive_mass": _summary(items, mass_key="raw_return_positive_mass"),
        }

    result = {
        "audit_entry": "phase3_heldout_segment_ablation_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": str(Path(args.train_suite_path).expanduser().resolve()),
            "heldout_suite_path": str(Path(args.heldout_suite_path).expanduser().resolve()),
            "reuse_zero_control_json": str(Path(args.reuse_zero_control_json).expanduser().resolve()),
            "reuse_pre_policy_json": str(Path(args.reuse_pre_policy_json).expanduser().resolve()),
            "reuse_heldout_subset_delta_json": (
                None
                if args.reuse_heldout_subset_delta_json is None
                else str(Path(args.reuse_heldout_subset_delta_json).expanduser().resolve())
            ),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "tail_len": int(args.tail_len),
            "runtime_wall_s": float(time.perf_counter() - t0),
        },
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "train_suite_summary": train_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "base_summary": base_summary,
        "ablation_summaries": ablation_summaries,
        "env_rows": env_rows,
        "conclusions": {
            "new_phase3_trusted_baseline_established": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
