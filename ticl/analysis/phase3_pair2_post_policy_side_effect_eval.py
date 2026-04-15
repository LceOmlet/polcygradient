import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg, _make_deterministic_batch_plan
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _default_device,
    _load_post_policy_bundle,
    _load_required_suite,
    _resolve_train_profile,
    _run_policy_eval,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH,
    PAIR2_BRANCH_SPECIFIC_MODE,
    PAIR2_CHECKPOINT_PATH,
    PAIR2_HELDOUT_SUITE_PATH,
    PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH,
    PAIR2_PREFLIGHT_JSON,
    PAIR2_TRAIN_SUITE_PATH,
    PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
    PAIR2_TRAIN_UPDATE_NSAMPLES,
    PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
    _load_pair2_preflight,
)
from ticl.analysis.prior_generalization_audit import _summarize_suite
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


def _delta_stats(values: list[float]) -> dict[str, Any]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "positive_count": int((finite > 0).sum().item()),
        "negative_count": int((finite < 0).sum().item()),
        "zero_count": int((finite == 0).sum().item()),
    }


def _per_env_deltas(*, baseline: dict[str, Any], override: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_full = [float(v) for v in baseline["full_return_per_env"]]
    override_full = [float(v) for v in override["full_return_per_env"]]
    baseline_suffix = [float(v) for v in baseline["suffix_return_per_env"]]
    override_suffix = [float(v) for v in override["suffix_return_per_env"]]
    if not (len(baseline_full) == len(override_full) == len(baseline_suffix) == len(override_suffix)):
        raise ValueError("Baseline/override per-env metric lengths do not match.")
    rows = []
    for idx, (base_full, over_full, base_suffix, over_suffix) in enumerate(
        zip(baseline_full, override_full, baseline_suffix, override_suffix)
    ):
        rows.append(
            {
                "env_index": int(idx),
                "baseline_full_return": base_full,
                "override_full_return": over_full,
                "full_return_delta": float(over_full - base_full),
                "baseline_suffix_return": base_suffix,
                "override_suffix_return": over_suffix,
                "suffix_return_delta": float(over_suffix - base_suffix),
            }
        )
    return rows


def _load_policy_and_eval(
    *,
    bundle_path: str,
    actor_objective_mode_override: str | None,
    suite_target: str,
    device_obj: torch.device,
) -> dict[str, Any]:
    checkpoint_path = str(Path(PAIR2_CHECKPOINT_PATH).expanduser().resolve())
    train_suite = _load_required_suite(PAIR2_TRAIN_SUITE_PATH)
    heldout_suite = _load_required_suite(PAIR2_HELDOUT_SUITE_PATH)

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    num_features = int(config["prior"]["num_features"])
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile="phase2_shared_backbone_contract",
        batch_size=256,
        n_epochs=1,
        learning_rate=2e-4,
        target_kl=0.03,
    )
    if actor_objective_mode_override is not None:
        profile_cfg["actor_objective_mode"] = str(actor_objective_mode_override).strip().lower()

    algo, _callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
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
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(list(train_suite["env_seeds"])[0])],
        rollout_rng_seeds=[int(list(train_suite["rollout_seeds"])[0])],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        actor_objective_runtime_current_suite_name=(
            "pair2" if actor_objective_mode_override is not None else None
        ),
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
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
    eval_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(
        PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS
    )
    train_suite_summary = _summarize_suite(eval_prior, train_suite)
    heldout_suite_summary = _summarize_suite(eval_prior, heldout_suite)
    bundle = _load_post_policy_bundle(
        bundle_path=str(bundle_path),
        checkpoint_path=checkpoint_path,
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(PAIR2_TRAIN_UPDATE_NSAMPLES),
        single_eval_pos=int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
        rollout_backend="serial",
        profile_cfg=profile_cfg,
        strict_native_rollout=True,
    )
    algo.policy.load_state_dict(bundle["policy_state_dict"], strict=True)

    target = str(suite_target).strip().lower()
    suite = heldout_suite if target == "heldout" else train_suite
    suite_summary = heldout_suite_summary if target == "heldout" else train_suite_summary
    t0 = time.perf_counter()
    metrics = _run_policy_eval(
        prior=eval_prior,
        suite=suite,
        policy=algo.policy,
        sampled=True,
        n_samples=int(PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES),
        num_features=int(num_features),
        single_eval_pos=int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
        device_obj=device_obj,
        rollout_backend="serial",
        deterministic_actor_sampling=True,
    )
    wall_s = float(time.perf_counter() - t0)
    vec_env.close()
    return {
        "bundle": {
            "source_path": str(bundle["source_path"]),
            "contract": dict(bundle["contract"]),
        },
        "suite_summary": suite_summary,
        "metrics": metrics,
        "wall_s": wall_s,
    }


def build_pair2_post_policy_side_effect_eval(*, suite_target: str = "heldout", device: str | None = None) -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    phase2_summary = assert_phase2_green(DEFAULT_PHASE2_SUMMARY)
    device_obj = torch.device(str(device or _default_device()))
    baseline_bundle_path = str(Path(PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH).expanduser().resolve())
    override_bundle_path = str(Path(PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH).expanduser().resolve())
    for path in (baseline_bundle_path, override_bundle_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing post-policy bundle {path}. Run phase3_pair2_train_update_ab_compare_pack.py with --overwrite first."
            )

    print(f"[pair2-side-effect] eval baseline bundle on {suite_target}", flush=True)
    baseline = _load_policy_and_eval(
        bundle_path=baseline_bundle_path,
        actor_objective_mode_override=None,
        suite_target=suite_target,
        device_obj=device_obj,
    )
    print(f"[pair2-side-effect] eval override bundle on {suite_target}", flush=True)
    override = _load_policy_and_eval(
        bundle_path=override_bundle_path,
        actor_objective_mode_override=PAIR2_BRANCH_SPECIFIC_MODE,
        suite_target=suite_target,
        device_obj=device_obj,
    )

    rows = _per_env_deltas(baseline=baseline["metrics"], override=override["metrics"])
    full_deltas = [float(row["full_return_delta"]) for row in rows]
    suffix_deltas = [float(row["suffix_return_delta"]) for row in rows]
    return {
        "audit_entry": "phase3_pair2_post_policy_side_effect_eval",
        "suite_target": str(suite_target).strip().lower(),
        "preflight": {
            "source_path": PAIR2_PREFLIGHT_JSON,
            "pair2_preflight_passed": bool(preflight.get("preflight_passed", False)),
            "pair2_unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
        },
        "contract": {
            "n_samples": int(PAIR2_TRAIN_UPDATE_NSAMPLES),
            "eval_n_samples": int(PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES),
            "single_eval_pos": int(PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS),
            "train_profile": "phase2_shared_backbone_contract",
            "strict_native_rollout": True,
            "baseline_actor_objective_mode": "tokenwise",
            "override_actor_objective_mode": PAIR2_BRANCH_SPECIFIC_MODE,
        },
        "baseline": baseline,
        "override": override,
        "comparison": {
            "override_minus_baseline_full_return_mean": float(
                override["metrics"]["full_return_mean"] - baseline["metrics"]["full_return_mean"]
            ),
            "override_minus_baseline_suffix_return_mean": float(
                override["metrics"]["suffix_return_mean"] - baseline["metrics"]["suffix_return_mean"]
            ),
            "full_return_delta_stats": _delta_stats(full_deltas),
            "suffix_return_delta_stats": _delta_stats(suffix_deltas),
            "per_env": rows,
        },
        "conclusions": {
            "eval_only": True,
            "no_train_loop": True,
            "override_has_non_target_side_effect": bool(
                abs(float(override["metrics"]["full_return_mean"] - baseline["metrics"]["full_return_mean"])) > 1e-6
                or abs(float(override["metrics"]["suffix_return_mean"] - baseline["metrics"]["suffix_return_mean"]))
                > 1e-6
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eval-only heldout/non-target side-effect check for saved pair2 A/B post-policy bundles."
    )
    parser.add_argument("--suite-target", type=str, default="heldout", choices=["heldout", "train"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_post_policy_side_effect_eval.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    os.environ["TICL_PHASE3_PROFILE_TIMING"] = "1"
    report = build_pair2_post_policy_side_effect_eval(
        suite_target=str(args.suite_target),
        device=args.device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
