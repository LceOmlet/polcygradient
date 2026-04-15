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
)
from ticl.analysis.phase3_subset_delta_fallback import resolve_heldout_subset_deltas
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


TAIL_LEN = 8
DIST_LIMIT = 4
LATE_K = 4


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


def _summary(items: list[dict[str, Any]], *, mass_key: str) -> dict[str, Any]:
    masses = [float(item[mass_key]) for item in items]
    terminal_resets = [float(item["objective_terminal_reset_count"]) for item in items]
    pre_gap = [float(item["pre_vs_zero_suffix_gap"]) for item in items]
    return {
        "corr_vs_terminal_resets": _corr(masses, terminal_resets),
        "corr_vs_pre_suffix_gap": _corr(masses, pre_gap),
        "mass_total": float(sum(masses)),
    }


def _late_dist4_tail_mass(adv_values: np.ndarray, segments: list[tuple[int, int]]) -> float:
    removed = 0.0
    for idx, (start, end) in enumerate(segments):
        if idx == 0:
            continue
        seg_adv = adv_values[start:end]
        seg_len = int(end - start)
        if seg_len <= 0:
            continue
        tail_start = max(0, seg_len - TAIL_LEN)
        for dist in range(1, int(DIST_LIMIT) + 1):
            pos = seg_len - dist
            if pos < 0 or pos >= seg_len:
                continue
            if pos < LATE_K:
                continue
            if pos < tail_start:
                continue
            val = float(seg_adv[pos])
            if val > 0.0:
                removed += val
    return removed


def main() -> int:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Phase 3 ablation: remove post-reset tail dist<=4 within late-step region (requires rollout)."
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

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )

    zero_reused = _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_zero_control_json),
        expected_policy_mode="zero",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
    )
    pre_reused = _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_pre_policy_json),
        expected_policy_mode="ppo",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
    )
    subset_deltas = resolve_heldout_subset_deltas(
        reuse_heldout_subset_delta_json=(
            None if args.reuse_heldout_subset_delta_json is None else str(args.reuse_heldout_subset_delta_json)
        ),
        zero_metrics=zero_reused,
        pre_metrics=pre_reused,
        reuse_zero_control_json=str(args.reuse_zero_control_json),
        reuse_pre_policy_json=str(args.reuse_pre_policy_json),
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend="serial",
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
    vec_env.close()

    env_rows: list[dict[str, Any]] = []
    for env_idx in range(int(objective_masks.shape[1])):
        obj = objective_masks[:, env_idx] > 1e-8
        if int(obj.sum()) <= 0:
            continue
        env_adv = actor_advantages[:, env_idx][obj]
        env_episode_starts = episode_starts[:, env_idx] > 1e-8
        segments = _objective_episode_segments(obj, env_episode_starts)
        removed_mass = _late_dist4_tail_mass(env_adv, segments)
        row = {
            "env_index": int(env_idx),
            "env_seed": int(list(heldout_suite["env_seeds"])[env_idx]),
            "rollout_seed": int(list(heldout_suite["rollout_seeds"])[env_idx]),
            "objective_token_count": int(obj.sum()),
            "objective_terminal_reset_count": int(_objective_terminal_reset_count_from_segments(segments)),
            "pre_vs_zero_suffix_gap": float(subset_deltas.get(env_idx, {}).get("pre_vs_zero_suffix_gap", 0.0)),
            "base_positive_mass": float(np.clip(env_adv, a_min=0.0, a_max=None).sum()),
            "removed_mass": float(removed_mass),
        }
        row["remaining_mass"] = float(max(0.0, row["base_positive_mass"] - row["removed_mass"]))
        env_rows.append(row)

    base_summary = _summary(env_rows, mass_key="base_positive_mass")
    ablated_summary = _summary(env_rows, mass_key="remaining_mass")

    result = {
        "audit_entry": "phase3_post_reset_tail_dist4_late_ablation_verify",
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
            "tail_len": int(TAIL_LEN),
            "dist_limit": int(DIST_LIMIT),
            "late_k": int(LATE_K),
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
        "env_rows": env_rows,
        "base_summary": base_summary,
        "post_reset_tail_dist4_late_ablation_summary": ablated_summary,
        "deltas": {
            "corr_reset_delta": float(
                ablated_summary["corr_vs_terminal_resets"] - base_summary["corr_vs_terminal_resets"]
            ),
            "corr_pre_delta": float(
                ablated_summary["corr_vs_pre_suffix_gap"] - base_summary["corr_vs_pre_suffix_gap"]
            ),
        },
        "conclusions": {
            "ablation_reduces_reset_bias": bool(
                ablated_summary["corr_vs_terminal_resets"] < base_summary["corr_vs_terminal_resets"]
            ),
            "ablation_improves_pre_gap_alignment": bool(
                ablated_summary["corr_vs_pre_suffix_gap"] > base_summary["corr_vs_pre_suffix_gap"]
            ),
            "new_phase3_trusted_baseline_established": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
