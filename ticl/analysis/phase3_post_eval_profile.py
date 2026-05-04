import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_guardrail import (
    TRUSTED_PHASE2_LAUNCH_SUMMARY,
    assert_phase2_launch_chain_green,
    assert_phase2_green,
)
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _default_device,
    _load_post_policy_bundle,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_ppo_policy_step_fn,
    _make_deterministic_batch_plan,
)
from ticl.analysis.prior_generalization_audit import (
    _resolve_audit_env_config,
    _summarize_suite,
    collect_suite_rewards_serial_profiled,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile Phase 3 post-train/post-heldout serial policy eval using a saved post-policy bundle."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--post-policy-bundle-path", type=str, required=True)
    parser.add_argument("--suite-target", type=str, choices=["train", "heldout"], required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--rollout-backend", type=str, default="serial", choices=["serial"])
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
    parser.add_argument("--phase2-summary-path", type=str, default=TRUSTED_PHASE2_LAUNCH_SUMMARY)
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    resolved_train_profile = str(args.train_profile).strip().lower()
    if resolved_train_profile == "trusted_sep_reset_mainline":
        phase2_summary = assert_phase2_launch_chain_green(args.phase2_summary_path)
    else:
        phase2_summary = assert_phase2_green(args.phase2_summary_path)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(args.device or _default_device()))

    train_suite = _load_required_suite(args.train_suite_path)
    heldout_suite = _load_required_suite(args.heldout_suite_path)

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(
        config,
        core_a=False,
        reference_semantics_enabled=False,
    )
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    num_features = int(config["prior"]["num_features"])
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=resolved_train_profile,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )
    strict_native_rollout = bool(
        optimizer_cfg.get(
            "ppo_strict_native_rollout",
            resolved_train_profile == "trusted_sep_reset_mainline",
        )
    )

    algo, _callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
        n_envs=1,
        n_steps=1,
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=1,
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
        strict_native_rollout=bool(strict_native_rollout),
        restore_validation_policy_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _make_deterministic_batch_plan(algo.rollout_buffer)

    eval_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    eval_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(args.single_eval_pos)
    train_suite_summary = _summarize_suite(eval_prior, train_suite)
    heldout_suite_summary = _summarize_suite(eval_prior, heldout_suite)
    bundle = _load_post_policy_bundle(
        bundle_path=str(args.post_policy_bundle_path),
        checkpoint_path=checkpoint_path,
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend=str(args.rollout_backend),
        profile_cfg=profile_cfg,
        strict_native_rollout=bool(strict_native_rollout),
    )
    algo.policy.load_state_dict(bundle["policy_state_dict"], strict=True)

    suite_target = str(args.suite_target).strip().lower()
    suite = train_suite if suite_target == "train" else heldout_suite
    suite_summary = train_suite_summary if suite_target == "train" else heldout_suite_summary
    policy_step_fn = _build_audit_ppo_policy_step_fn(
        algo.policy,
        sampled=True,
        single_eval_pos=int(args.single_eval_pos),
        boundary_contract_mode="normal",
    )

    t0 = time.perf_counter()
    profile = collect_suite_rewards_serial_profiled(
        eval_prior,
        suite=suite,
        policy_step_fn=policy_step_fn,
        n_samples=int(args.n_samples),
        num_features=int(num_features),
        single_eval_pos=int(args.single_eval_pos),
        device=device_obj,
        max_envs=args.max_envs,
    )
    wall_s = float(time.perf_counter() - t0)
    vec_env.close()

    report = {
        "audit_entry": "phase3_post_eval_profile",
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "config": {
            "checkpoint_path": checkpoint_path,
            "post_policy_bundle_path": str(Path(args.post_policy_bundle_path).expanduser().resolve()),
            "suite_target": suite_target,
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "rollout_backend": str(args.rollout_backend),
            "train_profile": str(profile_cfg["train_profile"]),
            "strict_native_rollout": bool(strict_native_rollout),
            "max_envs": None if args.max_envs is None else int(args.max_envs),
            "profile_build_n_envs": 1,
            "profile_build_n_steps": 1,
            "profile_build_batch_size": 1,
        },
        "suite_summary": suite_summary,
        "bundle_contract": dict(bundle["contract"]),
        "profile": profile,
        "wall_s": wall_s,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
