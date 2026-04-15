import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _load_required_suite,
    _resolve_train_profile,
)
from ticl.analysis.phase3_phase2_milestone_contract_gradient_compare import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_JSON as DEFAULT_MILESTONE_COMPARE_JSON,
    _default_device,
    _signature_diff,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import MaskedRecurrentRolloutBuffer, build_recurrent_ppo


DEFAULT_TRAIN_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt"
DEFAULT_HELDOUT_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt"
DEFAULT_OUTPUT_JSON = "/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json"


ALLOWED_PHASE3_BASELINE_DIFF_KEYS = {
    "n_envs",
    "env_rng_seeds",
    "rollout_rng_seeds",
}


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _build_many_env_signature(
    *,
    config: dict[str, Any],
    profile_cfg: dict[str, Any],
    train_suite: dict[str, Any],
    n_samples: int,
    strict_fixed_env_mode: bool,
    deterministic_actor_sampling: bool,
    deterministic_batch_plan: bool,
    strict_native_rollout: bool,
    actor_objective_runtime_current_suite_name: str | None,
) -> dict[str, Any]:
    optimizer_cfg = dict(config.get("optimizer", {}))
    return {
        "n_envs": int(train_suite["batch_size"]),
        "n_steps": int(n_samples),
        "learning_rate": float(profile_cfg["learning_rate"]),
        "batch_size": int(profile_cfg["batch_size"]),
        "n_epochs": int(profile_cfg["n_epochs"]),
        "gamma": float(optimizer_cfg.get("ppo_gamma", 1.0)),
        "gae_lambda": float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        "clip_range": optimizer_cfg.get("ppo_clip_range", 0.2),
        "normalize_advantage": bool(profile_cfg["normalize_advantage"]),
        "actor_gae_space": str(profile_cfg["actor_gae_space"]),
        "actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
        "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
        "reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
        "separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
        "strict_fixed_env_mode": bool(strict_fixed_env_mode),
        "env_rng_seeds": [int(v) for v in train_suite["env_seeds"]],
        "rollout_rng_seeds": [int(v) for v in train_suite["rollout_seeds"]],
        "deterministic_actor_sampling": bool(deterministic_actor_sampling),
        "deterministic_batch_plan": bool(deterministic_batch_plan),
        "strict_native_rollout": bool(strict_native_rollout),
        "restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
        "actor_objective_runtime_current_suite_name": actor_objective_runtime_current_suite_name,
        "runtime_normalized_q_value_weight_override": profile_cfg["runtime_normalized_q_value_weight_override"],
        "runtime_next_state_flow_matching_weight_override": profile_cfg["runtime_next_state_flow_matching_weight_override"],
        "ent_coef": float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        "vf_coef": float(profile_cfg["vf_coef"]),
        "max_grad_norm": float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        "target_kl": None if profile_cfg["target_kl"] is None else float(profile_cfg["target_kl"]),
    }


def _contract_diff_report(
    *,
    phase2_signature: dict[str, Any],
    phase3_signature: dict[str, Any],
    allowed_diff_keys: set[str],
) -> dict[str, Any]:
    diff = _signature_diff(phase2_signature, phase3_signature)
    unexpected = {
        key: value
        for key, value in diff.items()
        if key not in allowed_diff_keys
    }
    allowed = {
        key: value
        for key, value in diff.items()
        if key in allowed_diff_keys
    }
    return {
        "allowed_diff_keys": sorted(allowed_diff_keys),
        "all_diff": diff,
        "allowed_diff": allowed,
        "unexpected_diff": unexpected,
        "unexpected_diff_count": int(len(unexpected)),
        "unexpected_diff_free": bool(len(unexpected) == 0),
    }


def run_phase3_preflight_contract_check(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    heldout_suite_path: str,
    device: str | None = None,
    n_samples: int = 256,
    single_eval_pos: int = 64,
    batch_size: int | None = 256,
    n_epochs: int | None = 1,
    learning_rate: float | None = 2e-4,
    target_kl: float | None = 0.03,
    train_profile: str = "phase2_shared_backbone_contract",
    strict_fixed_env_mode: bool = True,
    deterministic_actor_sampling: bool = True,
    deterministic_batch_plan: bool = True,
    strict_native_rollout: bool = True,
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    milestone_compare_path: str = DEFAULT_MILESTONE_COMPARE_JSON,
) -> dict[str, Any]:
    wall_t0 = time.perf_counter()
    phase2_summary = assert_phase2_green(phase2_summary_path)
    milestone = _load_json(milestone_compare_path)
    if not bool(milestone.get("conclusions", {}).get("gradient_exact_match", False)):
        raise RuntimeError(
            f"Milestone compare is not green: {Path(milestone_compare_path).expanduser().resolve()}"
        )
    phase2_signature = dict(milestone["phase2_signature"])
    if phase2_signature.get("strict_native_rollout") is not True:
        raise RuntimeError("Guarded milestone signature must include strict_native_rollout=True.")

    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    heldout_suite_path = str(Path(heldout_suite_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    train_suite = _load_required_suite(train_suite_path)
    heldout_suite = _load_required_suite(heldout_suite_path)
    if int(train_suite["batch_size"]) != int(heldout_suite["batch_size"]):
        raise ValueError(
            "Preflight expects train and heldout suites to share batch_size. "
            f"Got train={int(train_suite['batch_size'])}, heldout={int(heldout_suite['batch_size'])}."
        )

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(train_suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )
    if str(profile_cfg["actor_objective_mode"]).strip().lower() != "tokenwise":
        raise ValueError(
            "This preflight checks the many-env baseline contract only; "
            f"got actor_objective_mode={profile_cfg['actor_objective_mode']!r}."
        )

    phase3_signature = _build_many_env_signature(
        config=config,
        profile_cfg=profile_cfg,
        train_suite=train_suite,
        n_samples=int(n_samples),
        strict_fixed_env_mode=bool(strict_fixed_env_mode),
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
        deterministic_batch_plan=bool(deterministic_batch_plan),
        strict_native_rollout=bool(strict_native_rollout),
        actor_objective_runtime_current_suite_name=None,
    )
    contract_diff = _contract_diff_report(
        phase2_signature=phase2_signature,
        phase3_signature=phase3_signature,
        allowed_diff_keys=ALLOWED_PHASE3_BASELINE_DIFF_KEYS,
    )

    algo = callback = vec_env = None
    try:
        algo, callback, vec_env = build_recurrent_ppo(
            model=model,
            env_prior=prior,
            device=str(device_obj),
            num_features=int(config["prior"]["num_features"]),
            n_envs=int(phase3_signature["n_envs"]),
            n_steps=int(phase3_signature["n_steps"]),
            learning_rate=float(phase3_signature["learning_rate"]),
            batch_size=int(phase3_signature["batch_size"]),
            n_epochs=int(phase3_signature["n_epochs"]),
            gamma=float(phase3_signature["gamma"]),
            gae_lambda=float(phase3_signature["gae_lambda"]),
            clip_range=phase3_signature["clip_range"],
            clip_range_vf=None,
            normalize_advantage=bool(phase3_signature["normalize_advantage"]),
            ent_coef=float(phase3_signature["ent_coef"]),
            vf_coef=float(phase3_signature["vf_coef"]),
            max_grad_norm=float(phase3_signature["max_grad_norm"]),
            target_kl=phase3_signature["target_kl"],
            reset_env_state_at_sep=bool(phase3_signature["reset_env_state_at_sep"]),
            separate_value_backbone=bool(phase3_signature["separate_value_backbone"]),
            strict_fixed_env_mode=bool(phase3_signature["strict_fixed_env_mode"]),
            env_rng_seeds=list(phase3_signature["env_rng_seeds"]),
            rollout_rng_seeds=list(phase3_signature["rollout_rng_seeds"]),
            deterministic_actor_sampling=bool(phase3_signature["deterministic_actor_sampling"]),
            deterministic_batch_plan=bool(phase3_signature["deterministic_batch_plan"]),
            strict_native_rollout=bool(phase3_signature["strict_native_rollout"]),
            restore_validation_policy_state=bool(phase3_signature["restore_validation_policy_state"]),
            actor_gae_space=str(phase3_signature["actor_gae_space"]),
            actor_baseline_mode=str(phase3_signature["actor_baseline_mode"]),
            actor_objective_mode=str(phase3_signature["actor_objective_mode"]),
            actor_objective_runtime_current_suite_name=phase3_signature[
                "actor_objective_runtime_current_suite_name"
            ],
            runtime_normalized_q_value_weight_override=phase3_signature[
                "runtime_normalized_q_value_weight_override"
            ],
            runtime_next_state_flow_matching_weight_override=phase3_signature[
                "runtime_next_state_flow_matching_weight_override"
            ],
            verbose=0,
        )
        del callback
        _bind_vec_env_to_fixed_suite(vec_env, suite=train_suite, single_eval_pos=int(single_eval_pos))
        if isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer):
            algo.rollout_buffer.set_deterministic_batch_plan(bool(deterministic_batch_plan))
        actor_core = algo.policy.rlpfn_model.rwkv_core
        value_model = getattr(algo.policy, "value_rlpfn_model", None)
        value_core_force_native = None
        if value_model is not None:
            value_core_force_native = bool(
                getattr(value_model.rwkv_core, "force_native_eval_forward_step", False)
            )
        built_runtime = {
            "algo_strict_fixed_env_mode": bool(getattr(algo, "_rwkv_strict_fixed_env_mode", False)),
            "algo_strict_native_rollout": bool(getattr(algo, "_rwkv_strict_native_rollout", False)),
            "algo_deterministic_actor_sampling": bool(
                getattr(algo, "_rwkv_deterministic_actor_sampling", False)
            ),
            "algo_deterministic_batch_plan": bool(getattr(algo, "_rwkv_deterministic_batch_plan", False)),
            "actor_core_force_native_eval_forward_step": bool(
                getattr(actor_core, "force_native_eval_forward_step", False)
            ),
            "value_core_force_native_eval_forward_step": value_core_force_native,
            "rollout_buffer_deterministic_batch_plan": bool(
                getattr(algo.rollout_buffer, "_deterministic_batch_plan", False)
            ),
            "rollout_buffer_actor_gae_space": str(getattr(algo.rollout_buffer, "_actor_gae_space", "")),
            "rollout_buffer_actor_baseline_mode": str(
                getattr(algo.rollout_buffer, "_actor_baseline_mode", "")
            ),
            "rollout_buffer_actor_objective_mode": str(
                getattr(algo.rollout_buffer, "_actor_objective_mode", "")
            ),
            "vec_env_num_envs": int(vec_env.num_envs),
            "vec_env_sample_seed_list": [int(v) for v in vec_env._sample_seed_list()],
        }
    finally:
        if vec_env is not None:
            vec_env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runtime_checks = {
        "phase2_guardrail_passed": bool(phase2_summary["validation"]["all_checks_pass"]),
        "milestone_gradient_exact_match": bool(milestone["conclusions"]["gradient_exact_match"]),
        "strict_native_rollout_enabled": bool(phase3_signature["strict_native_rollout"]),
        "build_strict_native_rollout_enabled": bool(built_runtime["algo_strict_native_rollout"]),
        "actor_core_force_native_enabled": bool(
            built_runtime["actor_core_force_native_eval_forward_step"]
        ),
        "strict_fixed_env_mode_enabled": bool(built_runtime["algo_strict_fixed_env_mode"]),
        "deterministic_actor_sampling_enabled": bool(
            built_runtime["algo_deterministic_actor_sampling"]
        ),
        "deterministic_batch_plan_enabled": bool(
            built_runtime["algo_deterministic_batch_plan"]
            and built_runtime["rollout_buffer_deterministic_batch_plan"]
        ),
        "suite_batch_matches_n_envs": int(train_suite["batch_size"]) == int(phase3_signature["n_envs"]),
        "suite_seed_lengths_match_n_envs": (
            len(train_suite["env_seeds"]) == int(phase3_signature["n_envs"])
            and len(train_suite["rollout_seeds"]) == int(phase3_signature["n_envs"])
        ),
        "vec_env_seed_binding_matches_suite": built_runtime["vec_env_sample_seed_list"]
        == [int(v) for v in train_suite["env_seeds"]],
        "unexpected_contract_diff_free": bool(contract_diff["unexpected_diff_free"]),
    }
    preflight_passed = bool(all(runtime_checks.values()))
    return {
        "audit_entry": "phase3_preflight_contract_check",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "elapsed_wall_time_sec": float(time.perf_counter() - wall_t0),
        "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
        "milestone_compare_path": str(Path(milestone_compare_path).expanduser().resolve()),
        "train_suite_path": train_suite_path,
        "heldout_suite_path": heldout_suite_path,
        "contract_args": {
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "batch_size": None if batch_size is None else int(batch_size),
            "n_epochs": None if n_epochs is None else int(n_epochs),
            "learning_rate": None if learning_rate is None else float(learning_rate),
            "target_kl": None if target_kl is None else float(target_kl),
            "train_profile": str(train_profile),
            "strict_fixed_env_mode": bool(strict_fixed_env_mode),
            "deterministic_actor_sampling": bool(deterministic_actor_sampling),
            "deterministic_batch_plan": bool(deterministic_batch_plan),
            "strict_native_rollout": bool(strict_native_rollout),
        },
        "phase2_signature": phase2_signature,
        "phase3_many_env_signature": phase3_signature,
        "contract_diff": contract_diff,
        "built_runtime": built_runtime,
        "runtime_checks": runtime_checks,
        "preflight_passed": preflight_passed,
        "allowed_next_differences": [
            "many-env n_envs/env_rng_seeds/rollout_rng_seeds",
            "exactly one explicitly declared narrow actor-objective adjustment in a later A/B",
        ],
        "forbidden_drift": [
            "actor-only substitution",
            "aux-flow/q-runtime override drift",
            "pooled weighting repair",
            "unguarded official eval shortcut in trusted compare",
            "normalize_advantage=True",
            "separate value backbone",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, nargs="?", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=DEFAULT_TRAIN_SUITE_PATH)
    parser.add_argument("--heldout-suite-path", type=str, default=DEFAULT_HELDOUT_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--train-profile", type=str, default="phase2_shared_backbone_contract")
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--milestone-compare-path", type=str, default=DEFAULT_MILESTONE_COMPARE_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_phase3_preflight_contract_check(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        heldout_suite_path=args.heldout_suite_path,
        device=args.device,
        n_samples=args.n_samples,
        single_eval_pos=args.single_eval_pos,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        train_profile=args.train_profile,
        phase2_summary_path=args.phase2_summary_path,
        milestone_compare_path=args.milestone_compare_path,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
