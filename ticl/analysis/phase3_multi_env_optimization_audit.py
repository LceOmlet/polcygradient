import argparse
import copy
import json
import os
import sys
import types
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_guardrail import (
    TRUSTED_PHASE2_LAUNCH_SUMMARY,
    assert_phase2_launch_chain_green,
    assert_phase2_green,
)
from ticl.analysis.critic_free_single_env_audit import (
    _audit_train_loop,
    _build_audit_env_cfg,
    _build_audit_ppo_policy_step_fn,
    _make_deterministic_batch_plan,
    _run_eval,
)
from ticl.analysis.prior_generalization_audit import (
    _build_zero_policy_step_fn,
    _maybe_load_suite,
    _resolve_audit_env_config,
    _summarize_suite,
    evaluate_prior_suite,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


def _phase3_profile_timing_enabled() -> bool:
    return str(os.environ.get("TICL_PHASE3_PROFILE_TIMING", "")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _phase3_profile_timing(label: str, started_at: float) -> None:
    if _phase3_profile_timing_enabled():
        print(f"[phase3-profile] {label}={time.perf_counter() - started_at:.3f}s", flush=True)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_required_suite(path: str) -> dict[str, Any]:
    suite = _maybe_load_suite(path)
    if suite is None:
        raise FileNotFoundError(f"Could not load suite: {path}")
    return suite


def _infer_runtime_scoped_suite_name_from_path(train_suite_path: str) -> str:
    resolved = str(Path(train_suite_path).expanduser().resolve()).lower()
    if "phase3_cross_env_baseline_pair2" in resolved:
        return "pair2"
    if "seed_13579" in resolved:
        return "seed13579"
    if "phase3_cross_env_baseline" in resolved:
        return "pair1"
    return str(Path(train_suite_path).expanduser().resolve().parent.name).strip().lower()


def _clone_suite_h_list(suite: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(h) for h in list(suite["h_list"])]


def _bind_vec_env_to_fixed_suite(vec_env, *, suite: dict[str, Any], single_eval_pos: int) -> None:
    suite_h_list_template = _clone_suite_h_list(suite)
    env_seeds = [int(v) for v in suite["env_seeds"]]
    rollout_seeds = [int(v) for v in suite["rollout_seeds"]]
    suite_batch_size = int(suite["batch_size"])
    if int(vec_env.num_envs) != suite_batch_size:
        raise ValueError(
            f"Suite batch_size={suite_batch_size} does not match vec_env.num_envs={int(vec_env.num_envs)}"
        )

    family_values = {
        vec_env.env_prior._normalize_family(h.get("family", "scm"))
        for h in suite_h_list_template
    }
    if len(family_values) != 1:
        raise RuntimeError(
            "Fixed-suite PPO training currently requires a fixed environment family across the batch. "
            f"Got families={sorted(family_values)}."
        )

    def _fixed_full_reset_batch(self, *, seeds=None):
        del seeds
        self._reset_seeds()
        self._h_list = [copy.deepcopy(h) for h in suite_h_list_template]
        self._env = self.env_prior._sample_environment_family_coarse_batch(
            self._h_list,
            device=self.device,
            rng_seeds=list(env_seeds),
            build_policy_generator=False,
        )
        self._rollout_generators = self.env_prior._make_generators_from_seeds(
            list(rollout_seeds),
            self.num_envs,
            self.device,
        )
        state_dim = int(self._env["state_dim"])
        zero_pad_dim = int(self._env["zero_pad_dim"])
        self._state_t = self.env_prior._stack_randn_with_generators(
            self._rollout_generators,
            (self.num_envs, state_dim),
            device=self.device,
            dtype=torch.float32,
        ) * self._env["init_state_std"][:, None]
        self._action_t = torch.zeros((self.num_envs, self.action_dim), device=self.device, dtype=torch.float32)
        init_action = self.env_prior._stack_randn_with_generators(
            self._rollout_generators,
            (self.num_envs, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        ) * self._env["init_action_std"][:, None]
        self._action_t.copy_(
            init_action * torch.as_tensor(self._vector_action_masks(), device=self.device, dtype=torch.float32)
        )
        self._reward_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._reward_mask_t = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        self._terminal_t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._terminal_signal_history = (
            torch.zeros((self.n_steps, self.num_envs), device=self.device, dtype=torch.float32)
            if bool(self._env["terminal_reset_enabled"].any().item())
            else None
        )
        self._zero_pad_t = torch.zeros((self.num_envs, zero_pad_dim), device=self.device, dtype=torch.float32)
        self._single_eval_pos = torch.full(
            (self.num_envs,),
            int(single_eval_pos),
            device=self.device,
            dtype=torch.long,
        )
        self._single_eval_pos_np = self._single_eval_pos.detach().cpu().numpy().astype(np.int64, copy=True)
        self._action_masks_np = None
        self._step_idx = 0
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self.reset_infos = [
            {
                "single_eval_pos": int(single_eval_pos),
                "env_seed": int(env_seeds[idx]),
                "rollout_seed": int(rollout_seeds[idx]),
            }
            for idx in range(self.num_envs)
        ]
        return self._build_obs_batch()

    vec_env._full_reset_batch = types.MethodType(_fixed_full_reset_batch, vec_env)
    vec_env._sample_seed_list = lambda: list(env_seeds)


def _run_policy_eval(
    *,
    prior: EnvironmentPrior,
    suite: dict[str, Any],
    policy,
    sampled: bool,
    n_samples: int,
    num_features: int,
    single_eval_pos: int,
    device_obj: torch.device,
    rollout_backend: str = "serial",
    deterministic_actor_sampling: bool = False,
):
    return _run_eval(
        prior=prior,
        suite=suite,
        policy_step_fn=_build_audit_ppo_policy_step_fn(
            policy,
            sampled=bool(sampled),
            single_eval_pos=int(single_eval_pos),
            boundary_contract_mode="normal",
            deterministic_actor_sampling=bool(deterministic_actor_sampling),
        ),
        n_samples=int(n_samples),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
        rollout_backend=str(rollout_backend),
    )


def _suite_gap(metrics: dict[str, Any], zero_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "full_return_gap": float(metrics["full_return_mean"] - zero_metrics["full_return_mean"]),
        "suffix_return_gap": float(metrics["suffix_return_mean"] - zero_metrics["suffix_return_mean"]),
    }


def _assert_reused_policy_mode(payload: dict[str, Any], *, expected_mode: str) -> None:
    expected_mode = str(expected_mode).strip().lower()
    policy_field = payload.get("policy", "")
    if isinstance(policy_field, dict):
        resolved_mode = str(policy_field.get("resolved_mode", "")).strip().lower()
        requested_mode = str(policy_field.get("requested_mode", "")).strip().lower()
        if resolved_mode != expected_mode and requested_mode != expected_mode:
            raise ValueError(f"Expected {expected_mode} policy artifact, got policy={policy_field!r}")
    else:
        if str(policy_field).strip().lower() != expected_mode:
            raise ValueError(f"Expected {expected_mode} policy artifact, got policy={policy_field!r}")


def _load_reused_suite_metrics(
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
    config = dict(payload.get("config", {}))
    _assert_reused_policy_mode(payload, expected_mode=str(expected_policy_mode))
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
                f"Zero-control artifact mismatch for {key}: observed={observed!r}, expected={expected_value!r}"
            )
    train_summary = dict(payload["train_suite"]["summary"])
    if train_summary.get("fingerprint") != train_suite_summary.get("fingerprint"):
        raise ValueError(
            "Train suite fingerprint mismatch for reused zero-control artifact: "
            f"{train_summary.get('fingerprint')!r} vs {train_suite_summary.get('fingerprint')!r}"
        )
    heldout_payload = payload.get("heldout_suite", None)
    if heldout_payload is not None:
        heldout_summary = dict(heldout_payload["summary"])
        if heldout_summary.get("fingerprint") != heldout_suite_summary.get("fingerprint"):
            raise ValueError(
                "Heldout suite fingerprint mismatch for reused zero-control artifact: "
                f"{heldout_summary.get('fingerprint')!r} vs {heldout_suite_summary.get('fingerprint')!r}"
            )
        heldout_metrics = dict(heldout_payload["metrics"])
    else:
        heldout_metrics = None
    return {
        "source_path": str(Path(report_json_path).expanduser().resolve()),
        "train": dict(payload["train_suite"]["metrics"]),
        "heldout": heldout_metrics,
    }


def _make_policy_bundle_contract(
    *,
    checkpoint_path: str,
    train_suite_summary: dict[str, Any],
    heldout_suite_summary: dict[str, Any],
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
    profile_cfg: dict[str, Any],
    strict_native_rollout: bool,
    restore_validation_policy_head_state: bool = False,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "train_suite_fingerprint": str(train_suite_summary.get("fingerprint")),
        "heldout_suite_fingerprint": str(heldout_suite_summary.get("fingerprint")),
        "n_samples": int(n_samples),
        "single_eval_pos": int(single_eval_pos),
        "rollout_backend": str(rollout_backend),
        "train_profile": str(profile_cfg["train_profile"]),
        "batch_size": int(profile_cfg["batch_size"]),
        "n_epochs": int(profile_cfg["n_epochs"]),
        "learning_rate": float(profile_cfg["learning_rate"]),
        "target_kl": None if profile_cfg["target_kl"] is None else float(profile_cfg["target_kl"]),
        "actor_objective_mode": str(profile_cfg["actor_objective_mode"]),
        "ppo_actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
        "ppo_normalize_advantage": bool(profile_cfg["normalize_advantage"]),
        "ppo_actor_gae_space": str(profile_cfg["actor_gae_space"]),
        "ppo_vf_coef": float(profile_cfg["vf_coef"]),
        "ppo_reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
        "ppo_separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
        "ppo_restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
        "ppo_restore_validation_policy_head_state": bool(restore_validation_policy_head_state),
        "strict_native_rollout": bool(strict_native_rollout),
    }


def _save_post_policy_bundle(
    *,
    bundle_path: str,
    algo,
    contract: dict[str, Any],
    train_history: dict[str, Any],
) -> str:
    resolved = str(Path(bundle_path).expanduser().resolve())
    path = Path(resolved)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_policy_state = {
        str(key): value.detach().cpu().clone()
        for key, value in algo.policy.state_dict().items()
    }
    torch.save(
        {
            "bundle_type": "phase3_post_policy_bundle",
            "contract": dict(contract),
            "train_history": copy.deepcopy(train_history),
            "policy_state_dict": cpu_policy_state,
        },
        path,
    )
    return resolved


def _load_post_policy_bundle(
    *,
    bundle_path: str,
    checkpoint_path: str,
    train_suite_summary: dict[str, Any],
    heldout_suite_summary: dict[str, Any],
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
    profile_cfg: dict[str, Any],
    strict_native_rollout: bool,
    restore_validation_policy_head_state: bool = False,
) -> dict[str, Any]:
    resolved = str(Path(bundle_path).expanduser().resolve())
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if str(payload.get("bundle_type", "")) != "phase3_post_policy_bundle":
        raise ValueError(f"Unexpected bundle type in {resolved}: {payload.get('bundle_type')!r}")
    observed_contract = dict(payload.get("contract", {}))
    expected_contract = _make_policy_bundle_contract(
        checkpoint_path=checkpoint_path,
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=n_samples,
        single_eval_pos=single_eval_pos,
        rollout_backend=rollout_backend,
        profile_cfg=profile_cfg,
        strict_native_rollout=bool(strict_native_rollout),
        restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
    )
    for key, expected_value in expected_contract.items():
        if key == "ppo_restore_validation_policy_head_state":
            observed_value = observed_contract.get(key, False)
        else:
            observed_value = observed_contract.get(key, None)
        if observed_value != expected_value:
            raise ValueError(
                f"Post-policy bundle mismatch for {key}: observed={observed_value!r}, expected={expected_value!r}"
            )
    policy_state_dict = payload.get("policy_state_dict", None)
    if not isinstance(policy_state_dict, dict) or not policy_state_dict:
        raise ValueError(f"Missing policy_state_dict in {resolved}")
    return {
        "source_path": resolved,
        "contract": observed_contract,
        "train_history": copy.deepcopy(payload.get("train_history", {})),
        "policy_state_dict": {
            str(key): value.detach().cpu().clone()
            for key, value in policy_state_dict.items()
        },
    }


def _resolve_train_profile(
    *,
    optimizer_cfg: dict[str, Any],
    train_profile: str,
    batch_size: int | None,
    n_epochs: int | None,
    learning_rate: float | None,
    target_kl: float | None,
) -> dict[str, Any]:
    profile = str(train_profile).strip().lower()
    if profile == "phase2_shared_backbone_contract":
        return {
            "train_profile": "phase2_shared_backbone_contract",
            "batch_size": int(batch_size) if batch_size is not None else 256,
            "n_epochs": int(n_epochs) if n_epochs is not None else 1,
            "learning_rate": float(learning_rate) if learning_rate is not None else 2e-4,
            "target_kl": float(target_kl) if target_kl is not None else 0.03,
            "normalize_advantage": False,
            "actor_gae_space": "raw",
            "actor_baseline_mode": "learned",
            "actor_objective_mode": "tokenwise",
            "reset_env_state_at_sep": True,
            "separate_value_backbone": False,
            "runtime_normalized_q_value_weight_override": None,
            "runtime_next_state_flow_matching_weight_override": None,
            "vf_coef": 0.5,
            "strict_native_rollout": False,
            "restore_validation_policy_state": False,
        }
    if profile == "trusted_sep_reset_mainline":
        return {
            "train_profile": "trusted_sep_reset_mainline",
            "batch_size": int(batch_size) if batch_size is not None else 256,
            "n_epochs": int(n_epochs) if n_epochs is not None else 1,
            "learning_rate": float(learning_rate) if learning_rate is not None else 2e-4,
            "target_kl": float(target_kl) if target_kl is not None else 0.03,
            "normalize_advantage": True,
            "actor_gae_space": "normalized",
            "actor_baseline_mode": "learned",
            "actor_objective_mode": "tokenwise",
            "reset_env_state_at_sep": True,
            "separate_value_backbone": False,
            "runtime_normalized_q_value_weight_override": None,
            "runtime_next_state_flow_matching_weight_override": None,
            "vf_coef": 0.5,
            "strict_native_rollout": True,
            "restore_validation_policy_state": True,
        }
    if profile == "legacy_actor_only_probe":
        return {
            "train_profile": "legacy_actor_only_probe",
            "batch_size": int(batch_size) if batch_size is not None else 2048,
            "n_epochs": int(n_epochs) if n_epochs is not None else 4,
            "learning_rate": float(learning_rate) if learning_rate is not None else 4e-4,
            "target_kl": None if target_kl is None else float(target_kl),
            "normalize_advantage": False,
            "actor_gae_space": "normalized",
            "actor_baseline_mode": "zero",
            "actor_objective_mode": "tokenwise",
            "reset_env_state_at_sep": False,
            "separate_value_backbone": False,
            "runtime_normalized_q_value_weight_override": 0.0,
            "runtime_next_state_flow_matching_weight_override": 0.0,
            "vf_coef": 0.0,
            "strict_native_rollout": True,
            "restore_validation_policy_state": True,
        }
    raise ValueError(f"Unsupported train_profile={train_profile!r}")


def _read_snapshot_target_match_count(snapshot_path: str | None) -> int | None:
    if snapshot_path in {None, ""}:
        return None
    resolved = Path(str(snapshot_path)).expanduser().resolve()
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if str(payload.get("snapshot_entry", "")) != "phase3_train_outer_batch_snapshot":
        raise RuntimeError(f"Unexpected train outer-batch snapshot payload at {resolved}")
    return int(payload.get("snapshot_target_match_count", 0))


def run_phase3_multi_env_optimization_audit(
    *,
    checkpoint_path: str,
    train_suite_path: str,
    heldout_suite_path: str,
    device: str | None = None,
    n_samples: int = 2048,
    single_eval_pos: int = 1946,
    batch_size: int | None = None,
    outer_epochs: int = 1,
    n_epochs: int | None = None,
    learning_rate: float | None = None,
    target_kl: float | None = None,
    rollout_backend: str = "serial",
    train_profile: str = "phase2_shared_backbone_contract",
    phase2_summary_path: str = TRUSTED_PHASE2_LAUNCH_SUMMARY,
    reuse_zero_control_json: str | None = None,
    reuse_pre_policy_json: str | None = None,
    save_post_policy_bundle_path: str | None = None,
    resume_post_policy_bundle_path: str | None = None,
    strict_fixed_env_mode: bool = False,
    deterministic_actor_sampling: bool = False,
    deterministic_batch_plan: bool = False,
    strict_native_rollout: bool = False,
    restore_validation_policy_head_state: bool = False,
    actor_objective_mode_override: str | None = None,
    actor_objective_runtime_current_suite_name: str | None = None,
    skip_heldout_eval: bool = False,
    eval_n_samples: int | None = None,
    train_outer_batch_snapshot_json: str | None = None,
    train_outer_batch_selector_trace_json: str | None = None,
    train_outer_batch_snapshot_target_outer_batch_idx: int | None = None,
    train_outer_batch_snapshot_target_env_index: int | None = None,
    train_outer_batch_snapshot_target_objective_episode_index: int | None = None,
    train_outer_batch_snapshot_target_objective_position_start: int | None = None,
    train_outer_batch_snapshot_target_objective_position_end: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    heldout_suite_path = str(Path(heldout_suite_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    resolved_train_profile = str(train_profile).strip().lower()
    if resolved_train_profile in {
        "phase2_shared_backbone_contract",
        "trusted_sep_reset_mainline",
    }:
        phase2_summary = assert_phase2_launch_chain_green(phase2_summary_path)
    else:
        phase2_summary = assert_phase2_green(phase2_summary_path)

    train_suite = _load_required_suite(train_suite_path)
    heldout_suite = _load_required_suite(heldout_suite_path)
    if int(train_suite["batch_size"]) != int(heldout_suite["batch_size"]):
        raise ValueError(
            "This optimization audit expects train and heldout suites to share batch_size. "
            f"Got train={int(train_suite['batch_size'])}, heldout={int(heldout_suite['batch_size'])}."
        )

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    if resolved_train_profile == "phase2_shared_backbone_contract":
        env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    else:
        env_cfg = _resolve_audit_env_config(
            config,
            core_a=False,
            reference_semantics_enabled=False,
        )
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(train_suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    num_features = int(config["prior"]["num_features"])
    train_batch_envs = int(train_suite["batch_size"])
    eval_n_samples_resolved = int(n_samples if eval_n_samples is None else eval_n_samples)
    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )
    if actor_objective_mode_override is not None:
        profile_cfg["actor_objective_mode"] = str(actor_objective_mode_override).strip().lower()

    resolved_actor_objective_runtime_current_suite_name = None
    if actor_objective_runtime_current_suite_name is not None:
        resolved_actor_objective_runtime_current_suite_name = (
            str(actor_objective_runtime_current_suite_name).strip().lower()
        )
    elif str(profile_cfg["actor_objective_mode"]).strip().lower() == "tokenwise_scale_env12_mid_episode_extension_block":
        resolved_actor_objective_runtime_current_suite_name = _infer_runtime_scoped_suite_name_from_path(
            train_suite_path
        )

    resolved_strict_native_rollout = bool(profile_cfg["strict_native_rollout"])
    if bool(strict_native_rollout):
        resolved_strict_native_rollout = True

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
        n_envs=int(train_batch_envs),
        n_steps=int(n_samples),
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
        strict_fixed_env_mode=bool(strict_fixed_env_mode),
        env_rng_seeds=[int(v) for v in train_suite["env_seeds"]] if bool(strict_fixed_env_mode) else None,
        rollout_rng_seeds=[int(v) for v in train_suite["rollout_seeds"]] if bool(strict_fixed_env_mode) else None,
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
        deterministic_batch_plan=bool(deterministic_batch_plan),
        strict_native_rollout=bool(resolved_strict_native_rollout),
        actor_objective_runtime_current_suite_name=resolved_actor_objective_runtime_current_suite_name,
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=bool(profile_cfg["restore_validation_policy_state"]),
        restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=train_suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo._rwkv_train_outer_batch_snapshot_path = (
        None
        if train_outer_batch_snapshot_json is None
        else str(Path(train_outer_batch_snapshot_json).expanduser().resolve())
    )
    algo._rwkv_train_outer_batch_snapshot_written = False
    algo._rwkv_last_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_selector_trace_path = (
        None
        if train_outer_batch_selector_trace_json is None
        else str(Path(train_outer_batch_selector_trace_json).expanduser().resolve())
    )
    algo._rwkv_train_outer_batch_selector_trace_rows = []
    algo._rwkv_train_outer_batch_selector_trace_completed = False
    algo._rwkv_last_train_outer_batch_selector_trace_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = (
        None
        if train_outer_batch_snapshot_target_outer_batch_idx is None
        else int(train_outer_batch_snapshot_target_outer_batch_idx)
    )
    algo._rwkv_train_outer_batch_snapshot_target_env_index = (
        None if train_outer_batch_snapshot_target_env_index is None else int(train_outer_batch_snapshot_target_env_index)
    )
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = (
        None
        if train_outer_batch_snapshot_target_objective_episode_index is None
        else int(train_outer_batch_snapshot_target_objective_episode_index)
    )
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = (
        None
        if train_outer_batch_snapshot_target_objective_position_start is None
        else int(train_outer_batch_snapshot_target_objective_position_start)
    )
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = (
        None
        if train_outer_batch_snapshot_target_objective_position_end is None
        else int(train_outer_batch_snapshot_target_objective_position_end)
    )

    eval_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    eval_prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    train_suite_summary = _summarize_suite(eval_prior, train_suite)
    heldout_suite_summary = _summarize_suite(eval_prior, heldout_suite)

    reused_zero_control = None
    if reuse_zero_control_json is not None:
        reused_zero_control = _load_reused_suite_metrics(
            report_json_path=str(reuse_zero_control_json),
            expected_policy_mode="zero",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(eval_n_samples_resolved),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )
        zero_train = dict(reused_zero_control["train"])
        zero_heldout = dict(reused_zero_control["heldout"])
    else:
        t_zero_train = time.perf_counter()
        zero_policy_step_fn = _build_zero_policy_step_fn()
        zero_train = evaluate_prior_suite(
            eval_prior,
            suite=train_suite,
            policy_step_fn=zero_policy_step_fn,
            n_samples=int(eval_n_samples_resolved),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
            rollout_backend=str(rollout_backend),
        )
        _phase3_profile_timing("zero_train_eval", t_zero_train)
        t_zero_heldout = time.perf_counter()
        zero_heldout = evaluate_prior_suite(
            eval_prior,
            suite=heldout_suite,
            policy_step_fn=zero_policy_step_fn,
            n_samples=int(eval_n_samples_resolved),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
            rollout_backend=str(rollout_backend),
        )
        _phase3_profile_timing("zero_heldout_eval", t_zero_heldout)

    reused_pre_policy = None
    if reuse_pre_policy_json is not None:
        reused_pre_policy = _load_reused_suite_metrics(
            report_json_path=str(reuse_pre_policy_json),
            expected_policy_mode="ppo",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(eval_n_samples_resolved),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )
        pre_train = dict(reused_pre_policy["train"])
        pre_heldout = None if reused_pre_policy["heldout"] is None else dict(reused_pre_policy["heldout"])
    else:
        t_pre_train = time.perf_counter()
        pre_train = _run_policy_eval(
            prior=eval_prior,
            suite=train_suite,
            policy=algo.policy,
            sampled=True,
            n_samples=eval_n_samples_resolved,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            device_obj=device_obj,
            rollout_backend=str(rollout_backend),
            deterministic_actor_sampling=bool(deterministic_actor_sampling),
        )
        _phase3_profile_timing("pre_train_eval", t_pre_train)
        if skip_heldout_eval:
            pre_heldout = None
        else:
            t_pre_heldout = time.perf_counter()
            pre_heldout = _run_policy_eval(
                prior=eval_prior,
                suite=heldout_suite,
                policy=algo.policy,
                sampled=True,
                n_samples=eval_n_samples_resolved,
                num_features=num_features,
                single_eval_pos=single_eval_pos,
                device_obj=device_obj,
                rollout_backend=str(rollout_backend),
                deterministic_actor_sampling=bool(deterministic_actor_sampling),
            )
            _phase3_profile_timing("pre_heldout_eval", t_pre_heldout)

    policy_bundle_contract = _make_policy_bundle_contract(
        checkpoint_path=checkpoint_path,
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        rollout_backend=str(rollout_backend),
        profile_cfg=profile_cfg,
        strict_native_rollout=bool(resolved_strict_native_rollout),
        restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
    )
    resumed_post_policy = None
    if resume_post_policy_bundle_path is not None:
        resumed_post_policy = _load_post_policy_bundle(
            bundle_path=str(resume_post_policy_bundle_path),
            checkpoint_path=checkpoint_path,
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
            profile_cfg=profile_cfg,
            strict_native_rollout=bool(resolved_strict_native_rollout),
            restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
        )
        algo.policy.load_state_dict(resumed_post_policy["policy_state_dict"], strict=True)
        train_history = copy.deepcopy(resumed_post_policy["train_history"])
    else:
        t_train = time.perf_counter()
        train_history = _audit_train_loop(
            algo,
            callback,
            vec_env,
            outer_epochs=int(outer_epochs),
        )
        _phase3_profile_timing("audit_train_loop", t_train)
        _make_deterministic_batch_plan(algo.rollout_buffer)
        if save_post_policy_bundle_path is not None:
            save_post_policy_bundle_path = _save_post_policy_bundle(
                bundle_path=str(save_post_policy_bundle_path),
                algo=algo,
                contract=policy_bundle_contract,
                train_history=train_history,
        )

    t_post_train = time.perf_counter()
    post_train = _run_policy_eval(
        prior=eval_prior,
        suite=train_suite,
        policy=algo.policy,
        sampled=True,
        n_samples=eval_n_samples_resolved,
        num_features=num_features,
        single_eval_pos=single_eval_pos,
        device_obj=device_obj,
        rollout_backend=str(rollout_backend),
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
    )
    _phase3_profile_timing("post_train_eval", t_post_train)
    if skip_heldout_eval:
        post_heldout = None
    else:
        t_post_heldout = time.perf_counter()
        post_heldout = _run_policy_eval(
            prior=eval_prior,
            suite=heldout_suite,
            policy=algo.policy,
            sampled=True,
            n_samples=eval_n_samples_resolved,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            device_obj=device_obj,
            rollout_backend=str(rollout_backend),
            deterministic_actor_sampling=bool(deterministic_actor_sampling),
        )
        _phase3_profile_timing("post_heldout_eval", t_post_heldout)

    vec_env.close()

    return {
        "audit_entry": "phase3_multi_env_optimization_audit",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "train_profile": str(profile_cfg["train_profile"]),
        "config": {
            "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "n_samples": int(n_samples),
            "eval_n_samples": int(eval_n_samples_resolved),
            "single_eval_pos": int(single_eval_pos),
            "train_suite_path": train_suite_path,
            "heldout_suite_path": heldout_suite_path,
            "train_batch_size": int(train_suite["batch_size"]),
            "heldout_batch_size": int(heldout_suite["batch_size"]),
            "rollout_backend": str(rollout_backend),
            "outer_epochs": int(outer_epochs),
            "n_epochs": int(profile_cfg["n_epochs"]),
            "batch_size": int(profile_cfg["batch_size"]),
            "learning_rate": float(profile_cfg["learning_rate"]),
            "target_kl": None if profile_cfg["target_kl"] is None else float(profile_cfg["target_kl"]),
            "ppo_actor_baseline_mode": str(profile_cfg["actor_baseline_mode"]),
            "ppo_normalize_advantage": bool(profile_cfg["normalize_advantage"]),
            "ppo_actor_gae_space": str(profile_cfg["actor_gae_space"]),
            "ppo_vf_coef": float(profile_cfg["vf_coef"]),
            "ppo_reset_env_state_at_sep": bool(profile_cfg["reset_env_state_at_sep"]),
            "ppo_separate_value_backbone": bool(profile_cfg["separate_value_backbone"]),
            "ppo_restore_validation_policy_state": bool(profile_cfg["restore_validation_policy_state"]),
            "ppo_restore_validation_policy_head_state": bool(restore_validation_policy_head_state),
            "strict_fixed_env_mode": bool(strict_fixed_env_mode),
            "deterministic_actor_sampling": bool(deterministic_actor_sampling),
            "deterministic_batch_plan": bool(deterministic_batch_plan),
            "strict_native_rollout": bool(resolved_strict_native_rollout),
            "actor_objective_mode_override": None
            if actor_objective_mode_override is None
            else str(actor_objective_mode_override).strip().lower(),
            "actor_objective_runtime_current_suite_name": resolved_actor_objective_runtime_current_suite_name,
            "runtime_normalized_q_value_weight_override": profile_cfg["runtime_normalized_q_value_weight_override"],
            "runtime_next_state_flow_matching_weight_override": profile_cfg["runtime_next_state_flow_matching_weight_override"],
            "reuse_zero_control_json": None
            if reuse_zero_control_json is None
            else str(Path(reuse_zero_control_json).expanduser().resolve()),
            "reuse_pre_policy_json": None
            if reuse_pre_policy_json is None
            else str(Path(reuse_pre_policy_json).expanduser().resolve()),
            "save_post_policy_bundle_path": None
            if save_post_policy_bundle_path is None
            else str(Path(save_post_policy_bundle_path).expanduser().resolve()),
            "resume_post_policy_bundle_path": None
            if resume_post_policy_bundle_path is None
            else str(Path(resume_post_policy_bundle_path).expanduser().resolve()),
            "train_outer_batch_snapshot_json": None
            if train_outer_batch_snapshot_json is None
            else str(Path(train_outer_batch_snapshot_json).expanduser().resolve()),
            "train_outer_batch_selector_trace_json": None
            if train_outer_batch_selector_trace_json is None
            else str(Path(train_outer_batch_selector_trace_json).expanduser().resolve()),
            "train_outer_batch_snapshot_target_outer_batch_idx": None
            if train_outer_batch_snapshot_target_outer_batch_idx is None
            else int(train_outer_batch_snapshot_target_outer_batch_idx),
            "train_outer_batch_snapshot_target_env_index": None
            if train_outer_batch_snapshot_target_env_index is None
            else int(train_outer_batch_snapshot_target_env_index),
            "train_outer_batch_snapshot_target_objective_episode_index": None
            if train_outer_batch_snapshot_target_objective_episode_index is None
            else int(train_outer_batch_snapshot_target_objective_episode_index),
            "train_outer_batch_snapshot_target_objective_position_start": None
            if train_outer_batch_snapshot_target_objective_position_start is None
            else int(train_outer_batch_snapshot_target_objective_position_start),
            "train_outer_batch_snapshot_target_objective_position_end": None
            if train_outer_batch_snapshot_target_objective_position_end is None
            else int(train_outer_batch_snapshot_target_objective_position_end),
        },
        "phase2_preflight": {
            "required": True,
            "summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "phase2_pass_flags": dict(phase2_summary["pass_flags"]),
            "launch_chain_locked": bool(
                resolved_train_profile in {
                    "phase2_shared_backbone_contract",
                    "trusted_sep_reset_mainline",
                }
            ),
            "launch_chain_source_of_truth": None
            if not isinstance(phase2_summary.get("source_of_truth", None), dict)
            else dict(phase2_summary["source_of_truth"]),
        },
        "train_suite_summary": train_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "zero_control": {
            "train": zero_train,
            "heldout": zero_heldout,
            "reused": bool(reused_zero_control is not None),
            "source_path": None if reused_zero_control is None else str(reused_zero_control["source_path"]),
        },
        "pre": {
            "train": pre_train,
            "heldout": pre_heldout,
            "reused": bool(reused_pre_policy is not None),
            "source_path": None if reused_pre_policy is None else str(reused_pre_policy["source_path"]),
        },
        "post": {
            "train": post_train,
            "heldout": post_heldout,
        },
        "post_policy_bundle": {
            "saved": bool(save_post_policy_bundle_path is not None and resumed_post_policy is None),
            "save_path": None if resumed_post_policy is not None else save_post_policy_bundle_path,
            "resumed": bool(resumed_post_policy is not None),
            "source_path": None if resumed_post_policy is None else str(resumed_post_policy["source_path"]),
        },
        "train_outer_batch_snapshot": {
            "requested_path": None
            if train_outer_batch_snapshot_json is None
            else str(Path(train_outer_batch_snapshot_json).expanduser().resolve()),
            "selector": {
                "target_outer_batch_idx": None
                if train_outer_batch_snapshot_target_outer_batch_idx is None
                else int(train_outer_batch_snapshot_target_outer_batch_idx),
                "target_env_index": None
                if train_outer_batch_snapshot_target_env_index is None
                else int(train_outer_batch_snapshot_target_env_index),
                "target_objective_episode_index": None
                if train_outer_batch_snapshot_target_objective_episode_index is None
                else int(train_outer_batch_snapshot_target_objective_episode_index),
                "target_objective_position_start": None
                if train_outer_batch_snapshot_target_objective_position_start is None
                else int(train_outer_batch_snapshot_target_objective_position_start),
                "target_objective_position_end": None
                if train_outer_batch_snapshot_target_objective_position_end is None
                else int(train_outer_batch_snapshot_target_objective_position_end),
            },
            "written_path": getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None),
            "written": bool(getattr(algo, "_rwkv_train_outer_batch_snapshot_written", False)),
            "snapshot_target_match_count": _read_snapshot_target_match_count(
                getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None)
            ),
        },
        "train_outer_batch_selector_trace": {
            "requested_path": None
            if train_outer_batch_selector_trace_json is None
            else str(Path(train_outer_batch_selector_trace_json).expanduser().resolve()),
            "selector": {
                "target_outer_batch_idx": None
                if train_outer_batch_snapshot_target_outer_batch_idx is None
                else int(train_outer_batch_snapshot_target_outer_batch_idx),
                "target_env_index": None
                if train_outer_batch_snapshot_target_env_index is None
                else int(train_outer_batch_snapshot_target_env_index),
                "target_objective_episode_index": None
                if train_outer_batch_snapshot_target_objective_episode_index is None
                else int(train_outer_batch_snapshot_target_objective_episode_index),
                "target_objective_position_start": None
                if train_outer_batch_snapshot_target_objective_position_start is None
                else int(train_outer_batch_snapshot_target_objective_position_start),
                "target_objective_position_end": None
                if train_outer_batch_snapshot_target_objective_position_end is None
                else int(train_outer_batch_snapshot_target_objective_position_end),
            },
            "written_path": getattr(algo, "_rwkv_last_train_outer_batch_selector_trace_path", None),
            "completed": bool(getattr(algo, "_rwkv_train_outer_batch_selector_trace_completed", False)),
            "row_count": int(len(getattr(algo, "_rwkv_train_outer_batch_selector_trace_rows", []) or [])),
        },
        "comparison": {
            "pre_train_vs_zero": _suite_gap(pre_train, zero_train),
            "pre_heldout_vs_zero": None if pre_heldout is None else _suite_gap(pre_heldout, zero_heldout),
            "post_train_vs_zero": _suite_gap(post_train, zero_train),
            "post_heldout_vs_zero": None if post_heldout is None else _suite_gap(post_heldout, zero_heldout),
            "train_full_return_delta": float(post_train["full_return_mean"] - pre_train["full_return_mean"]),
            "train_suffix_return_delta": float(post_train["suffix_return_mean"] - pre_train["suffix_return_mean"]),
            "heldout_full_return_delta": None
            if (post_heldout is None or pre_heldout is None)
            else float(post_heldout["full_return_mean"] - pre_heldout["full_return_mean"]),
            "heldout_suffix_return_delta": None
            if (post_heldout is None or pre_heldout is None)
            else float(post_heldout["suffix_return_mean"] - pre_heldout["suffix_return_mean"]),
        },
        "train_history": train_history,
        "heldout_eval_skipped": bool(skip_heldout_eval),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 multi-env optimization audit: fixed train-suite PPO training, fixed train/heldout-suite evaluation."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--outer-epochs", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--rollout-backend", type=str, default="serial", choices=["serial", "family_vectorized"])
    parser.add_argument(
        "--train-profile",
        type=str,
        default="phase2_shared_backbone_contract",
        choices=["phase2_shared_backbone_contract", "trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--phase2-summary-path", type=str, default=TRUSTED_PHASE2_LAUNCH_SUMMARY)
    parser.add_argument("--reuse-zero-control-json", type=str, default=None)
    parser.add_argument("--reuse-pre-policy-json", type=str, default=None)
    parser.add_argument("--save-post-policy-bundle-path", type=str, default=None)
    parser.add_argument("--resume-post-policy-bundle-path", type=str, default=None)
    parser.add_argument("--strict-fixed-env-mode", action="store_true")
    parser.add_argument("--deterministic-actor-sampling", action="store_true")
    parser.add_argument("--deterministic-batch-plan", action="store_true")
    parser.add_argument("--strict-native-rollout", action="store_true")
    parser.add_argument("--actor-objective-mode-override", type=str, default=None)
    parser.add_argument("--actor-objective-runtime-current-suite-name", type=str, default=None)
    parser.add_argument("--train-outer-batch-snapshot-json", type=str, default=None)
    parser.add_argument("--train-outer-batch-selector-trace-json", type=str, default=None)
    parser.add_argument("--train-outer-batch-snapshot-target-outer-batch-idx", type=int, default=None)
    parser.add_argument("--train-outer-batch-snapshot-target-env-index", type=int, default=None)
    parser.add_argument("--train-outer-batch-snapshot-target-objective-episode-index", type=int, default=None)
    parser.add_argument("--train-outer-batch-snapshot-target-objective-position-start", type=int, default=None)
    parser.add_argument("--train-outer-batch-snapshot-target-objective-position-end", type=int, default=None)
    parser.add_argument("--skip-heldout-eval", action="store_true")
    parser.add_argument("--eval-n-samples", type=int, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    report = run_phase3_multi_env_optimization_audit(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        heldout_suite_path=args.heldout_suite_path,
        device=args.device,
        n_samples=args.n_samples,
        single_eval_pos=args.single_eval_pos,
        batch_size=args.batch_size,
        outer_epochs=args.outer_epochs,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        rollout_backend=args.rollout_backend,
        train_profile=args.train_profile,
        phase2_summary_path=args.phase2_summary_path,
        reuse_zero_control_json=args.reuse_zero_control_json,
        reuse_pre_policy_json=args.reuse_pre_policy_json,
        save_post_policy_bundle_path=args.save_post_policy_bundle_path,
        resume_post_policy_bundle_path=args.resume_post_policy_bundle_path,
        strict_fixed_env_mode=bool(args.strict_fixed_env_mode),
        deterministic_actor_sampling=bool(args.deterministic_actor_sampling),
        deterministic_batch_plan=bool(args.deterministic_batch_plan),
        strict_native_rollout=bool(args.strict_native_rollout),
        actor_objective_mode_override=args.actor_objective_mode_override,
        actor_objective_runtime_current_suite_name=args.actor_objective_runtime_current_suite_name,
        skip_heldout_eval=bool(args.skip_heldout_eval),
        eval_n_samples=args.eval_n_samples,
        train_outer_batch_snapshot_json=args.train_outer_batch_snapshot_json,
        train_outer_batch_selector_trace_json=args.train_outer_batch_selector_trace_json,
        train_outer_batch_snapshot_target_outer_batch_idx=args.train_outer_batch_snapshot_target_outer_batch_idx,
        train_outer_batch_snapshot_target_env_index=args.train_outer_batch_snapshot_target_env_index,
        train_outer_batch_snapshot_target_objective_episode_index=args.train_outer_batch_snapshot_target_objective_episode_index,
        train_outer_batch_snapshot_target_objective_position_start=args.train_outer_batch_snapshot_target_objective_position_start,
        train_outer_batch_snapshot_target_objective_position_end=args.train_outer_batch_snapshot_target_objective_position_end,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
