import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
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


def _load_reused_per_env_metrics(
    *,
    report_json_path: str,
    expected_policy_mode: str,
    train_suite_summary: dict[str, Any],
    heldout_suite_summary: dict[str, Any],
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
) -> dict[str, Any]:
    payload = json.loads(Path(report_json_path).expanduser().resolve().read_text())
    policy_field = payload.get("policy", "")
    expected_mode = str(expected_policy_mode).strip().lower()
    if isinstance(policy_field, dict):
        resolved_mode = str(policy_field.get("resolved_mode", "")).strip().lower()
        requested_mode = str(policy_field.get("requested_mode", "")).strip().lower()
        if resolved_mode != expected_mode and requested_mode != expected_mode:
            raise ValueError(f"Unexpected policy artifact mode in {report_json_path}: {policy_field!r}")
    else:
        if str(policy_field).strip().lower() != expected_mode:
            raise ValueError(f"Unexpected policy artifact mode in {report_json_path}: {policy_field!r}")
    config = dict(payload.get("config", {}))
    expected = {
        "n_samples": int(n_samples),
        "single_eval_pos": int(single_eval_pos),
        "rollout_backend": str(rollout_backend),
        "core_a": False,
        "reference_semantics_enabled": False,
    }
    for key, expected_value in expected.items():
        observed = config.get(key, None)
        if observed != expected_value:
            raise ValueError(
                f"Reused artifact mismatch for {key}: observed={observed!r}, expected={expected_value!r}"
            )

    train_summary = dict(payload["train_suite"]["summary"])
    heldout_summary = dict(payload["heldout_suite"]["summary"])
    if train_summary.get("fingerprint") != train_suite_summary.get("fingerprint"):
        raise ValueError("Train suite fingerprint mismatch for reused artifact")
    if heldout_summary.get("fingerprint") != heldout_suite_summary.get("fingerprint"):
        raise ValueError("Heldout suite fingerprint mismatch for reused artifact")

    out = {}
    for key in ("train_suite", "heldout_suite"):
        metrics = dict(payload[key]["metrics"])
        for required in ("full_return_per_env", "suffix_return_per_env"):
            if required not in metrics:
                raise ValueError(f"Missing {required} in reused artifact {report_json_path}")
        out["train" if key == "train_suite" else "heldout"] = {
            "full_return_per_env": [float(v) for v in metrics["full_return_per_env"]],
            "suffix_return_per_env": [float(v) for v in metrics["suffix_return_per_env"]],
        }
    return {
        "source_path": str(Path(report_json_path).expanduser().resolve()),
        **out,
    }


def _single_env_suite(suite: dict, env_index: int) -> dict:
    idx = int(env_index)
    return {
        "h_list": [copy.deepcopy(list(suite["h_list"])[idx])],
        "env_seeds": [int(list(suite["env_seeds"])[idx])],
        "rollout_seeds": [int(list(suite["rollout_seeds"])[idx])],
        "batch_size": 1,
        "suite_seed": suite.get("suite_seed", None),
    }


def _compute_concentration(values: list[float]) -> dict[str, Any]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) <= 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "positive_count": 0,
            "negative_count": 0,
            "nonpositive_count": 0,
            "abs_mass_sum": 0.0,
            "top1_abs_mass_share": float("nan"),
            "top2_abs_mass_share": float("nan"),
            "abs_mass_hhi": float("nan"),
        }
    abs_vals = finite.abs()
    abs_sum = float(abs_vals.sum().item())
    if abs_sum > 0:
        shares = abs_vals / abs_vals.sum()
        sorted_shares, _ = torch.sort(shares, descending=True)
        top1 = float(sorted_shares[0].item())
        top2 = float(sorted_shares[: min(2, int(sorted_shares.numel()))].sum().item())
        hhi = float((shares ** 2).sum().item())
    else:
        top1 = 0.0
        top2 = 0.0
        hhi = 0.0
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if int(finite.numel()) > 1 else 0.0,
        "positive_count": int((finite > 0).sum().item()),
        "negative_count": int((finite < 0).sum().item()),
        "nonpositive_count": int((finite <= 0).sum().item()),
        "abs_mass_sum": abs_sum,
        "top1_abs_mass_share": top1,
        "top2_abs_mass_share": top2,
        "abs_mass_hhi": hhi,
    }


def _evaluate_subset_post_policy(
    *,
    prior: EnvironmentPrior,
    suite: dict[str, Any],
    policy_step_fn,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device: torch.device,
    max_envs: int,
) -> dict[str, Any]:
    prof = collect_suite_rewards_serial_profiled(
        prior,
        suite=suite,
        policy_step_fn=policy_step_fn,
        n_samples=int(n_samples),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device,
        max_envs=int(max_envs),
    )
    env_metrics = []
    for item in prof["env_profiles"]:
        env_metrics.append(
            {
                "env_index": int(item["env_index"]),
                "env_seed": int(item["env_seed"]),
                "rollout_seed": int(item["rollout_seed"]),
                "full_return": float(item["full_return_sum"]),
                "suffix_return": float(item["suffix_return_sum"]),
                "rollout_s": float(item["rollout_s"]),
            }
        )
    return {
        "profile": prof,
        "env_metrics": env_metrics,
    }


def _combine_pre_zero_post(
    *,
    subset_eval: dict[str, Any],
    pre_per_env: list[float],
    zero_per_env: list[float],
) -> dict[str, Any]:
    combined = []
    full_delta = []
    suffix_delta = []
    full_gap_delta = []
    suffix_gap_delta = []
    for item in subset_eval["env_metrics"]:
        idx = int(item["env_index"])
        pre_full = float(pre_per_env[idx])
        zero_full = float(zero_per_env[idx])
        pre_suffix = float(pre_per_env[idx]) if False else None
        # suffix handled separately below
        combined_item = {
            "env_index": idx,
            "env_seed": int(item["env_seed"]),
            "rollout_seed": int(item["rollout_seed"]),
            "post_full_return": float(item["full_return"]),
            "pre_full_return": pre_full,
            "zero_full_return": zero_full,
            "post_vs_zero_full_gap": float(item["full_return"] - zero_full),
            "pre_vs_zero_full_gap": float(pre_full - zero_full),
            "full_return_delta": float(item["full_return"] - pre_full),
        }
        combined.append(combined_item)
        full_delta.append(combined_item["full_return_delta"])
        full_gap_delta.append(combined_item["post_vs_zero_full_gap"] - combined_item["pre_vs_zero_full_gap"])
    return {
        "env_items": combined,
        "full_return_delta_stats": _compute_concentration(full_delta),
        "full_gap_delta_stats": _compute_concentration(full_gap_delta),
    }


def _combine_suffix(
    *,
    env_items: list[dict[str, Any]],
    pre_suffix_per_env: list[float],
    zero_suffix_per_env: list[float],
    post_suffix_per_env: list[float],
) -> dict[str, Any]:
    suffix_delta = []
    suffix_gap_delta = []
    for item, post_suffix in zip(env_items, post_suffix_per_env):
        idx = int(item["env_index"])
        pre_suffix = float(pre_suffix_per_env[idx])
        zero_suffix = float(zero_suffix_per_env[idx])
        item["post_suffix_return"] = float(post_suffix)
        item["pre_suffix_return"] = pre_suffix
        item["zero_suffix_return"] = zero_suffix
        item["post_vs_zero_suffix_gap"] = float(post_suffix - zero_suffix)
        item["pre_vs_zero_suffix_gap"] = float(pre_suffix - zero_suffix)
        item["suffix_return_delta"] = float(post_suffix - pre_suffix)
        suffix_delta.append(item["suffix_return_delta"])
        suffix_gap_delta.append(item["post_vs_zero_suffix_gap"] - item["pre_vs_zero_suffix_gap"])
    return {
        "suffix_return_delta_stats": _compute_concentration(suffix_delta),
        "suffix_gap_delta_stats": _compute_concentration(suffix_gap_delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight Phase 3 per-environment delta concentration probe using reused zero/pre artifacts and a saved post-policy bundle."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--reuse-zero-control-json", type=str, required=True)
    parser.add_argument("--reuse-pre-policy-json", type=str, required=True)
    parser.add_argument("--post-policy-bundle-path", type=str, required=True)
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
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--max-envs", type=int, default=4)
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
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
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
        rollout_backend=str(args.rollout_backend),
    )
    reused_pre = _load_reused_per_env_metrics(
        report_json_path=str(args.reuse_pre_policy_json),
        expected_policy_mode="ppo",
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend=str(args.rollout_backend),
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
    bundle = _load_post_policy_bundle(
        bundle_path=str(args.post_policy_bundle_path),
        checkpoint_path=checkpoint_path,
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend=str(args.rollout_backend),
        profile_cfg=profile_cfg,
    )
    algo.policy.load_state_dict(bundle["policy_state_dict"], strict=True)
    policy_step_fn = _build_audit_ppo_policy_step_fn(
        algo.policy,
        sampled=True,
        single_eval_pos=int(args.single_eval_pos),
        boundary_contract_mode="normal",
    )

    max_envs = int(args.max_envs)
    train_eval = _evaluate_subset_post_policy(
        prior=eval_prior,
        suite=train_suite,
        policy_step_fn=policy_step_fn,
        n_samples=int(args.n_samples),
        num_features=int(num_features),
        single_eval_pos=int(args.single_eval_pos),
        device=device_obj,
        max_envs=max_envs,
    )
    heldout_eval = _evaluate_subset_post_policy(
        prior=eval_prior,
        suite=heldout_suite,
        policy_step_fn=policy_step_fn,
        n_samples=int(args.n_samples),
        num_features=int(num_features),
        single_eval_pos=int(args.single_eval_pos),
        device=device_obj,
        max_envs=max_envs,
    )
    vec_env.close()

    train_combined = _combine_pre_zero_post(
        subset_eval=train_eval,
        pre_per_env=reused_pre["train"]["full_return_per_env"],
        zero_per_env=reused_zero["train"]["full_return_per_env"],
    )
    train_combined.update(
        _combine_suffix(
            env_items=train_combined["env_items"],
            pre_suffix_per_env=reused_pre["train"]["suffix_return_per_env"],
            zero_suffix_per_env=reused_zero["train"]["suffix_return_per_env"],
            post_suffix_per_env=[float(v["suffix_return"]) for v in train_eval["env_metrics"]],
        )
    )
    heldout_combined = _combine_pre_zero_post(
        subset_eval=heldout_eval,
        pre_per_env=reused_pre["heldout"]["full_return_per_env"],
        zero_per_env=reused_zero["heldout"]["full_return_per_env"],
    )
    heldout_combined.update(
        _combine_suffix(
            env_items=heldout_combined["env_items"],
            pre_suffix_per_env=reused_pre["heldout"]["suffix_return_per_env"],
            zero_suffix_per_env=reused_zero["heldout"]["suffix_return_per_env"],
            post_suffix_per_env=[float(v["suffix_return"]) for v in heldout_eval["env_metrics"]],
        )
    )

    report = {
        "audit_entry": "phase3_env_delta_concentration_probe",
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
        },
        "config": {
            "checkpoint_path": checkpoint_path,
            "reuse_zero_control_json": str(Path(args.reuse_zero_control_json).expanduser().resolve()),
            "reuse_pre_policy_json": str(Path(args.reuse_pre_policy_json).expanduser().resolve()),
            "post_policy_bundle_path": str(Path(args.post_policy_bundle_path).expanduser().resolve()),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "rollout_backend": str(args.rollout_backend),
            "train_profile": str(profile_cfg["train_profile"]),
            "max_envs": max_envs,
            "profile_build_n_envs": 1,
            "profile_build_n_steps": 1,
            "profile_build_batch_size": 1,
        },
        "bundle_contract": dict(bundle["contract"]),
        "train_suite_summary": train_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "train_subset": {
            "post_eval_profile": train_eval["profile"],
            "combined": train_combined,
        },
        "heldout_subset": {
            "post_eval_profile": heldout_eval["profile"],
            "combined": heldout_combined,
        },
        "conclusions": {
            "new_phase3_trusted_baseline_established": False,
            "subset_only_diagnostic": True,
            "train_suffix_delta_top1_abs_mass_share": float(train_combined["suffix_return_delta_stats"]["top1_abs_mass_share"]),
            "heldout_suffix_delta_top1_abs_mass_share": float(heldout_combined["suffix_return_delta_stats"]["top1_abs_mass_share"]),
            "train_suffix_gap_delta_mean": float(train_combined["suffix_gap_delta_stats"]["mean"]),
            "heldout_suffix_gap_delta_mean": float(heldout_combined["suffix_gap_delta_stats"]["mean"]),
            "train_suffix_gap_delta_positive_count": int(train_combined["suffix_gap_delta_stats"]["positive_count"]),
            "heldout_suffix_gap_delta_positive_count": int(heldout_combined["suffix_gap_delta_stats"]["positive_count"]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
