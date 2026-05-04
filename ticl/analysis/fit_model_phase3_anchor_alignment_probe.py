import argparse
import copy
import json
import random
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ticl.rlpfn_skyline_env import apply_rlpfn_skyline_env_defaults

apply_rlpfn_skyline_env_defaults(["rlpfn"])

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase3_longrun_gradient_compare import (
    _build_fixed_env_h,
    _collect_and_measure_total_grad,
    _default_device,
    _fixed_env_contract_summary,
    _resolve_continue_run_config,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_anchor_contract import apply_rlpfn_anchor_compare_contract
from ticl.rl_validation import evaluate_rlpfn_on_gym_envs
from ticl.sb3_recurrent_ppo import (
    build_recurrent_ppo,
    build_validation_recurrent_ppo_policy,
    extract_validation_recurrent_ppo_policy_state,
)
from ticl.train import _resolve_recurrent_ppo_training_bridge_kwargs


DEFAULT_CHECKPOINT_PATH = "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"
DEFAULT_OUTPUT_JSON = "/home/chen/RLPFN/artifacts/fit_model_phase3_anchor_alignment_probe.json"


def _clone_h_repeated(h, count: int) -> list[dict[str, Any]]:
    return [copy.deepcopy(h) for _ in range(int(count))]


def _max_abs_diff_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().to(dtype=torch.float32)
    b = b.detach().cpu().to(dtype=torch.float32)
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if int(a.numel()) == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _tensor_compare_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return {
        "shape_left": [int(v) for v in left.shape],
        "shape_right": [int(v) for v in right.shape],
        "exact_match": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "max_abs_diff": float(_max_abs_diff_tensor(left, right)),
    }


def _build_fit_model_anchor_compare_config(
    *,
    checkpoint_path: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    train_env_seed: int,
    rollout_seed: int,
) -> dict[str, Any]:
    config = _resolve_continue_run_config(checkpoint_path=checkpoint_path)
    optimizer_cfg = config["optimizer"]
    optimizer_cfg["ppo_n_steps"] = int(n_steps)
    optimizer_cfg["ppo_batch_size"] = int(batch_size)
    apply_rlpfn_anchor_compare_contract(
        config,
        learning_rate=float(learning_rate),
        env_rng_seeds=[int(train_env_seed)],
        rollout_rng_seeds=[int(rollout_seed)],
    )
    return config


def _build_active_fit_model_signature(config: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = dict(config.get("optimizer", {}))
    prior_env = dict(config.get("prior", {}).get("environment", {}))
    q_runtime_override = optimizer_cfg.get("ppo_runtime_normalized_q_value_weight_override")
    flow_runtime_override = optimizer_cfg.get("ppo_runtime_next_state_flow_matching_weight_override")
    effective_q_weight = (
        q_runtime_override if q_runtime_override is not None else prior_env.get("normalized_q_value_weight")
    )
    effective_flow_weight = (
        flow_runtime_override
        if flow_runtime_override is not None
        else prior_env.get("next_state_flow_matching_weight")
    )
    return {
        "optimizer.rl_objective": optimizer_cfg.get("rl_objective"),
        "optimizer.learning_rate": optimizer_cfg.get("learning_rate"),
        "optimizer.ppo_n_epochs": optimizer_cfg.get("ppo_n_epochs"),
        "optimizer.ppo_gamma": optimizer_cfg.get("ppo_gamma"),
        "optimizer.ppo_gae_lambda": optimizer_cfg.get("ppo_gae_lambda"),
        "optimizer.ppo_clip_range": optimizer_cfg.get("ppo_clip_range"),
        "optimizer.ppo_normalize_advantage": optimizer_cfg.get("ppo_normalize_advantage"),
        "optimizer.ppo_actor_gae_space": optimizer_cfg.get("ppo_actor_gae_space"),
        "optimizer.ppo_actor_baseline_mode": optimizer_cfg.get("ppo_actor_baseline_mode"),
        "optimizer.ppo_separate_value_backbone": optimizer_cfg.get("ppo_separate_value_backbone"),
        "optimizer.ppo_reset_env_state_at_sep": optimizer_cfg.get("ppo_reset_env_state_at_sep"),
        "optimizer.ppo_restore_validation_policy_state": optimizer_cfg.get(
            "ppo_restore_validation_policy_state"
        ),
        "optimizer.ppo_strict_native_rollout": optimizer_cfg.get("ppo_strict_native_rollout"),
        "optimizer.ppo_strict_fixed_env_mode": optimizer_cfg.get("ppo_strict_fixed_env_mode"),
        "optimizer.ppo_env_rng_seeds": optimizer_cfg.get("ppo_env_rng_seeds"),
        "optimizer.ppo_rollout_rng_seeds": optimizer_cfg.get("ppo_rollout_rng_seeds"),
        "optimizer.ppo_deterministic_actor_sampling": optimizer_cfg.get(
            "ppo_deterministic_actor_sampling"
        ),
        "optimizer.ppo_deterministic_batch_plan": optimizer_cfg.get("ppo_deterministic_batch_plan"),
        "optimizer.ppo_runtime_normalized_q_value_weight_override": q_runtime_override,
        "optimizer.ppo_runtime_next_state_flow_matching_weight_override": flow_runtime_override,
        "optimizer.ppo_vf_coef": optimizer_cfg.get("ppo_vf_coef"),
        "optimizer.ppo_target_kl": optimizer_cfg.get("ppo_target_kl"),
        "prior.environment.normalized_q_value_weight": prior_env.get("normalized_q_value_weight"),
        "prior.environment.next_state_flow_matching_weight": prior_env.get("next_state_flow_matching_weight"),
        "effective.normalized_q_value_weight": effective_q_weight,
        "effective.next_state_flow_matching_weight": effective_flow_weight,
    }


def _signature_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted(set(left.keys()) | set(right.keys())):
        if left.get(key) != right.get(key):
            out[str(key)] = {"left": left.get(key), "right": right.get(key)}
    return out


class _ValidationActionRewardEnv:
    def __init__(self, action_log, *, obs_dim=4, action_dim=1, rollout_lengths=(2, 2)):
        self._action_log = action_log
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._rollout_lengths = [int(v) for v in rollout_lengths]
        self._rollout_idx = -1
        self._step_idx = 0
        self.action_space = types.SimpleNamespace(
            low=-np.ones((self._action_dim,), dtype=np.float32),
            high=np.ones((self._action_dim,), dtype=np.float32),
            shape=(self._action_dim,),
        )

    def reset(self, seed=None):
        del seed
        self._rollout_idx += 1
        self._step_idx = 0
        obs = np.full((self._obs_dim,), float(self._rollout_idx), dtype=np.float32)
        return obs, {}

    def step(self, action):
        action_scalar = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._action_log.append(float(action_scalar))
        self._step_idx += 1
        rollout_idx = min(self._rollout_idx, len(self._rollout_lengths) - 1)
        terminated = bool(self._step_idx >= self._rollout_lengths[rollout_idx])
        obs = np.full(
            (self._obs_dim,),
            float(self._rollout_idx) + 0.1 * float(self._step_idx),
            dtype=np.float32,
        )
        reward = float(action_scalar)
        return obs, reward, terminated, False, {}

    def close(self):
        return None


class _FakeSyncVectorEnv:
    def __init__(self, env_fns, copy=True, observation_mode="same"):
        del observation_mode
        self.copy = bool(copy)
        self.envs = [fn() for fn in env_fns]
        self.num_envs = int(len(self.envs))
        self._observations = None
        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is None:
            seed = [None] * self.num_envs
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        observations = []
        infos = {}
        for idx, (env, env_seed) in enumerate(zip(self.envs, seed)):
            obs, info = env.reset(seed=env_seed)
            observations.append(np.asarray(obs, dtype=np.float32))
            for key, value in dict(info).items():
                infos.setdefault(key, np.empty((self.num_envs,), dtype=object))
                infos.setdefault(f"_{key}", np.zeros((self.num_envs,), dtype=np.bool_))
                infos[key][idx] = value
                infos[f"_{key}"][idx] = True
        self._observations = np.stack(observations, axis=0)
        self._rewards.fill(0.0)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._autoreset_envs.fill(False)
        return np.copy(self._observations) if self.copy else self._observations, infos

    def step(self, actions):
        observations = []
        infos = {}
        for idx, (env, action) in enumerate(zip(self.envs, np.asarray(actions))):
            if self._autoreset_envs[idx]:
                obs, info = env.reset(seed=None)
                reward = 0.0
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32))
            self._rewards[idx] = float(reward)
            self._terminations[idx] = bool(terminated)
            self._truncations[idx] = bool(truncated)
            for key, value in dict(info).items():
                infos.setdefault(key, np.empty((self.num_envs,), dtype=object))
                infos.setdefault(f"_{key}", np.zeros((self.num_envs,), dtype=np.bool_))
                infos[key][idx] = value
                infos[f"_{key}"][idx] = True
        self._observations = np.stack(observations, axis=0)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)
        return (
            np.copy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


@contextmanager
def _patched_validation_gym(*, action_log):
    import gymnasium as gym

    original_make = getattr(gym, "make")
    original_vector = getattr(gym, "vector", None)

    def _env_factory(_env_name):
        return _ValidationActionRewardEnv(action_log)

    gym.make = _env_factory
    gym.vector = types.SimpleNamespace(SyncVectorEnv=_FakeSyncVectorEnv)
    try:
        yield
    finally:
        gym.make = original_make
        if original_vector is None:
            try:
                delattr(gym, "vector")
            except Exception:
                pass
        else:
            gym.vector = original_vector


def _attach_validation_handles(target, *, live_policy, saved_policy_state):
    if target is None:
        return
    if live_policy is None:
        target.__dict__.pop("_validation_sb3_policy_live", None)
    else:
        target.__dict__["_validation_sb3_policy_live"] = live_policy
    target.__dict__["_validation_ppo_policy_state"] = saved_policy_state


def _validation_path_result_summary(*, mean_return, per_env, action_log):
    env_metrics = dict(per_env["DummyEnv-vValidationE2E"])
    return {
        "mean_return": float(mean_return),
        "env_return_mean": float(env_metrics["return_mean"]),
        "env_return": float(env_metrics.get("return", env_metrics["return_mean"])),
        "env_len_mean": float(env_metrics["len_mean"]),
        "action_log": [float(v) for v in action_log],
    }


def _validation_pair_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_actions = np.asarray(left["action_log"], dtype=np.float32)
    right_actions = np.asarray(right["action_log"], dtype=np.float32)
    if left_actions.shape != right_actions.shape:
        exact_action_log_match = False
        max_abs_action_diff = float("inf")
    else:
        exact_action_log_match = bool(np.array_equal(left_actions, right_actions))
        max_abs_action_diff = (
            0.0 if left_actions.size == 0 else float(np.max(np.abs(left_actions - right_actions)))
        )
    return {
        "mean_return_diff": float(left["mean_return"] - right["mean_return"]),
        "env_return_mean_diff": float(left["env_return_mean"] - right["env_return_mean"]),
        "env_return_diff": float(left["env_return"] - right["env_return"]),
        "env_len_mean_diff": float(left["env_len_mean"] - right["env_len_mean"]),
        "exact_action_log_match": bool(exact_action_log_match),
        "max_abs_action_log_diff": float(max_abs_action_diff),
    }


def _run_validation_e2e_compare(*, built: dict[str, Any], configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for mode_name in ("fit_model_active_bridge", "phase3_anchor_reference"):
        built_mode = built[mode_name]
        resolved_config = copy.deepcopy(configs[mode_name])
        resolved_config["orchestration"] = dict(resolved_config.get("orchestration", {}))
        resolved_config["orchestration"].update(
            {
                "rl_validate_envs": "DummyEnv-vValidationE2E",
                "rl_validate_episodes": 2,
                "rl_validate_max_steps": 8,
                "rl_validate_seed": 7,
                "rl_validate_context_lower_bound": 1,
                "rl_validate_max_parallel_columns": 2,
            }
        )
        saved_policy_state = extract_validation_recurrent_ppo_policy_state(built_mode["algo"].policy)
        for validation_path, live_policy in (
            ("live_policy", built_mode["algo"].policy),
            ("saved_policy_rebuild", None),
        ):
            action_log = []
            with _patched_validation_gym(action_log=action_log):
                for target in (
                    built_mode["model"],
                    getattr(built_mode["model"], "module", None),
                    getattr(built_mode["model"], "_orig_mod", None),
                ):
                    _attach_validation_handles(
                        target,
                        live_policy=live_policy,
                        saved_policy_state=saved_policy_state,
                    )
                mean_return, per_env = evaluate_rlpfn_on_gym_envs(
                    model=built_mode["model"],
                    config=resolved_config,
                )
            summaries[f"{mode_name}.{validation_path}"] = _validation_path_result_summary(
                mean_return=mean_return,
                per_env=per_env,
                action_log=action_log,
            )

    pairwise = {
        "fit_model_live_vs_phase3_live": _validation_pair_summary(
            summaries["fit_model_active_bridge.live_policy"],
            summaries["phase3_anchor_reference.live_policy"],
        ),
        "fit_model_rebuild_vs_phase3_rebuild": _validation_pair_summary(
            summaries["fit_model_active_bridge.saved_policy_rebuild"],
            summaries["phase3_anchor_reference.saved_policy_rebuild"],
        ),
        "fit_model_live_vs_rebuild": _validation_pair_summary(
            summaries["fit_model_active_bridge.live_policy"],
            summaries["fit_model_active_bridge.saved_policy_rebuild"],
        ),
        "phase3_live_vs_rebuild": _validation_pair_summary(
            summaries["phase3_anchor_reference.live_policy"],
            summaries["phase3_anchor_reference.saved_policy_rebuild"],
        ),
    }
    all_equal = all(
        float(summary["mean_return_diff"]) == 0.0
        and float(summary["env_return_mean_diff"]) == 0.0
        and float(summary["env_return_diff"]) == 0.0
        and float(summary["env_len_mean_diff"]) == 0.0
        and bool(summary["exact_action_log_match"])
        and float(summary["max_abs_action_log_diff"]) == 0.0
        for summary in pairwise.values()
    )
    return {
        "paths": summaries,
        "pairwise": pairwise,
        "conclusion": {
            "all_return_paths_exact_match": bool(all_equal),
            "all_action_logs_exact_match": bool(all_equal),
            "no_validation_semantic_diff_detected": bool(all_equal),
        },
    }


def _build_algo_from_fit_model_config(
    *,
    checkpoint_path: str,
    resolved_config: dict[str, Any],
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    load_model.cache_clear()
    model, _ = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(resolved_config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(resolved_config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    optimizer_cfg = resolved_config["optimizer"]
    build_kwargs, runtime_summary = _resolve_recurrent_ppo_training_bridge_kwargs(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=1,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(optimizer_cfg.get("ppo_n_epochs", 4)),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=optimizer_cfg.get("ppo_clip_range_vf", None),
        normalize_advantage=bool(optimizer_cfg.get("ppo_normalize_advantage", True)),
        actor_gae_space=str(optimizer_cfg.get("ppo_actor_gae_space", "normalized")),
        actor_baseline_mode=str(optimizer_cfg.get("ppo_actor_baseline_mode", "learned")),
        separate_value_backbone=bool(optimizer_cfg.get("ppo_separate_value_backbone", False)),
        reset_env_state_at_sep=bool(optimizer_cfg.get("ppo_reset_env_state_at_sep", True)),
        runtime_normalized_q_value_weight_override=optimizer_cfg.get(
            "ppo_runtime_normalized_q_value_weight_override", None
        ),
        runtime_next_state_flow_matching_weight_override=optimizer_cfg.get(
            "ppo_runtime_next_state_flow_matching_weight_override", None
        ),
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(optimizer_cfg.get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=optimizer_cfg.get("ppo_target_kl", None),
        verbose=0,
        ppo_restore_validation_policy_state=optimizer_cfg.get(
            "ppo_restore_validation_policy_state", None
        ),
        ppo_strict_fixed_env_mode=optimizer_cfg.get("ppo_strict_fixed_env_mode", False),
        ppo_env_rng_seeds=optimizer_cfg.get("ppo_env_rng_seeds", None),
        ppo_rollout_rng_seeds=optimizer_cfg.get("ppo_rollout_rng_seeds", None),
        ppo_deterministic_actor_sampling=optimizer_cfg.get(
            "ppo_deterministic_actor_sampling", False
        ),
        ppo_deterministic_batch_plan=optimizer_cfg.get("ppo_deterministic_batch_plan", False),
        ppo_strict_native_rollout=optimizer_cfg.get("ppo_strict_native_rollout", False),
    )
    algo, callback, vec_env = build_recurrent_ppo(**build_kwargs)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(n_steps))

    validation_policy = build_validation_recurrent_ppo_policy(
        model=model,
        env_cfg=env_cfg,
        device=device_obj,
        num_features=int(num_features),
        policy_state_dict=_resolve_saved_policy_state(model),
        strict_native_rollout=bool(optimizer_cfg.get("ppo_strict_native_rollout", False)),
    )
    return {
        "algo": algo,
        "callback": callback,
        "vec_env": vec_env,
        "model": model,
        "env_cfg": env_cfg,
        "runtime_summary": runtime_summary,
        "validation_policy": validation_policy,
    }


def _resolve_saved_policy_state(model):
    state = getattr(model, "_validation_ppo_policy_state", None)
    return state if isinstance(state, dict) else None


def _build_fixed_validation_inputs(*, env_cfg: dict[str, Any], num_features: int, device) -> dict[str, Any]:
    obs_slot_dim = int(env_cfg.get("obs_slot_dim", 400))
    action_slot_dim = int(env_cfg.get("action_slot_dim", 30))
    terminal_enabled = bool(env_cfg.get("terminal_reset_enabled", False))
    obs_t = torch.linspace(-0.4, 0.4, steps=min(11, int(obs_slot_dim)), device=device, dtype=torch.float32).reshape(1, -1)
    obs_dim = int(obs_t.shape[-1])
    action_t = torch.linspace(-0.2, 0.2, steps=int(action_slot_dim), device=device, dtype=torch.float32).reshape(1, -1)
    reward_t = torch.tensor([[0.125]], device=device, dtype=torch.float32)
    reward_mask_t = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    env_info = {
        "obs_slot_dim": int(obs_slot_dim),
        "action_slot_dim": int(action_slot_dim),
        "action_dim": int(action_slot_dim),
        "action_dim_per_sample": torch.tensor([int(action_slot_dim)], device=device, dtype=torch.long),
        "obs_dim": torch.tensor([int(obs_dim)], device=device, dtype=torch.long),
        "terminal_reset_enabled": torch.tensor([bool(terminal_enabled)], device=device, dtype=torch.bool),
        "phase_t": torch.zeros((1, 1), device=device, dtype=torch.float32),
    }
    if terminal_enabled:
        env_info["terminal_t"] = torch.zeros((1, 1), device=device, dtype=torch.float32)
    return {
        "obs_t": obs_t,
        "action_t": action_t,
        "reward_t": reward_t,
        "reward_mask_t": reward_mask_t,
        "cache": None,
        "step_idx": 0,
        "env_info": env_info,
        "num_features": int(num_features),
    }


def _run_validation_step_compare(policy_a, policy_b, *, env_cfg: dict[str, Any], num_features: int, device) -> dict[str, Any]:
    inputs = _build_fixed_validation_inputs(env_cfg=env_cfg, num_features=num_features, device=device)
    step_a = policy_a.make_vectorized_rollout_step_fn()
    step_b = policy_b.make_vectorized_rollout_step_fn()
    with torch.no_grad():
        out_a, cache_a = step_a(
            inputs["obs_t"],
            inputs["action_t"],
            inputs["reward_t"],
            inputs["reward_mask_t"],
            inputs["cache"],
            inputs["step_idx"],
            inputs["env_info"],
        )
        out_b, cache_b = step_b(
            inputs["obs_t"],
            inputs["action_t"],
            inputs["reward_t"],
            inputs["reward_mask_t"],
            inputs["cache"],
            inputs["step_idx"],
            inputs["env_info"],
        )
    del cache_a, cache_b
    return {
        "action_mean": _tensor_compare_summary(out_a["action_mean"], out_b["action_mean"]),
        "action_std": _tensor_compare_summary(out_a["action_std"], out_b["action_std"]),
        "values": _tensor_compare_summary(out_a["values"], out_b["values"]),
        "value_logits": _tensor_compare_summary(out_a["value_logits"], out_b["value_logits"]),
    }


def _localize(report: dict[str, Any]) -> dict[str, Any]:
    grad_cosine = float(report["training_compare"]["grad_compare"]["cosine_similarity"])
    grad_l2 = float(report["training_compare"]["grad_compare"]["l2_delta_norm"])
    saved_head_diff = float(report["training_compare"]["saved_policy_state_compare"]["max_abs_diff"])
    validation_action_diff = float(report["validation_compare"]["action_mean"]["max_abs_diff"])
    validation_value_diff = float(report["validation_compare"]["value_logits"]["max_abs_diff"])
    validation_e2e_ok = bool(
        report.get("validation_e2e_compare", {})
        .get("conclusion", {})
        .get("no_validation_semantic_diff_detected", False)
    )
    runtime = dict(report["fit_model_runtime"])

    if (
        bool(runtime["checkpoint_saved_validation_policy_state_present"])
        and not bool(runtime["restore_validation_policy_state"])
        and saved_head_diff > 0.0
    ):
        stage = "unrestored_validation_policy_state"
        reason = (
            "fit_model training build saw a checkpoint-saved PPO validation policy state but did not restore it "
            "before exporting the live validation handles."
        )
    elif bool(runtime["strict_native_rollout"]) is False and (validation_action_diff > 0.0 or validation_value_diff > 0.0):
        stage = "validation_forward_path_drift"
        reason = (
            "policy heads line up, but validation still diverges because the fit_model PPO bridge is not forcing the "
            "native RWKV eval-forward path used by the anchor contract."
        )
    elif not bool(validation_e2e_ok):
        stage = "validation_e2e_semantic_drift"
        reason = (
            "single-step validation outputs line up, but the public evaluate_rlpfn_on_gym_envs path still diverges "
            "between live-policy and/or rebuilt-saved-policy validation routes."
        )
    elif grad_cosine < 0.9999 or grad_l2 > 1e-6:
        stage = "training_gradient_drift"
        reason = "training-side rollout or optimization bridge still differs from the explicit phase3 anchor contract."
    else:
        stage = "aligned"
        reason = "fit_model PPO training build and validation bridge match the phase3 anchor contract on the fixed compare setup."
    return {
        "stage": stage,
        "reason": reason,
        "grad_cosine_similarity": grad_cosine,
        "grad_l2_delta_norm": grad_l2,
        "saved_policy_state_max_abs_diff": saved_head_diff,
        "validation_action_mean_max_abs_diff": validation_action_diff,
        "validation_value_logits_max_abs_diff": validation_value_diff,
    }


def run_fit_model_phase3_anchor_alignment_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    rollout_seed: int = 2020,
    single_eval_pos: int = 1946,
    n_steps: int = 2048,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    fit_model_config = _build_fit_model_anchor_compare_config(
        checkpoint_path=checkpoint_path,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        train_env_seed=int(train_env_seed),
        rollout_seed=int(rollout_seed),
    )
    anchor_config = copy.deepcopy(fit_model_config)
    apply_rlpfn_anchor_compare_contract(
        anchor_config,
        learning_rate=float(learning_rate),
        env_rng_seeds=[int(train_env_seed)],
        rollout_rng_seeds=[int(rollout_seed)],
    )

    fit_model_sig = _build_active_fit_model_signature(fit_model_config)
    anchor_sig = _build_active_fit_model_signature(anchor_config)
    sig_diff = _signature_diff(fit_model_sig, anchor_sig)

    env_cfg = _build_audit_env_cfg(fit_model_config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    reports = {}
    grad_vectors = {}
    built = {}
    for mode_name, resolved_config in (
        ("fit_model_active_bridge", fit_model_config),
        ("phase3_anchor_reference", anchor_config),
    ):
        random.seed(int(rollout_seed))
        np.random.seed(int(rollout_seed))
        torch.manual_seed(int(rollout_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rollout_seed))
        started_at = time.perf_counter()
        built_mode = _build_algo_from_fit_model_config(
            checkpoint_path=checkpoint_path,
            resolved_config=resolved_config,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            rollout_seed=int(rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )
        built_mode["build_wall_s"] = float(time.perf_counter() - started_at)
        try:
            report = _collect_and_measure_total_grad(
                built_mode["algo"],
                built_mode["callback"],
                built_mode["vec_env"],
            )
            grad_vectors[mode_name] = report.pop("grad_vector")
            reports[mode_name] = report
            built[mode_name] = built_mode
        finally:
            pass

    grad_a = grad_vectors["fit_model_active_bridge"]
    grad_b = grad_vectors["phase3_anchor_reference"]
    cosine = torch.nn.functional.cosine_similarity(
        grad_a.unsqueeze(0),
        grad_b.unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).item()
    grad_delta = grad_b - grad_a

    fit_model_policy_state = extract_validation_recurrent_ppo_policy_state(
        built["fit_model_active_bridge"]["algo"].policy
    )
    anchor_policy_state = extract_validation_recurrent_ppo_policy_state(
        built["phase3_anchor_reference"]["algo"].policy
    )
    saved_policy_state = _resolve_saved_policy_state(built["fit_model_active_bridge"]["model"])
    saved_state_compare = _tensor_compare_summary(
        fit_model_policy_state["action_net.weight"],
        saved_policy_state["action_net.weight"],
    )
    validation_compare = _run_validation_step_compare(
        built["fit_model_active_bridge"]["algo"].policy,
        built["fit_model_active_bridge"]["validation_policy"],
        env_cfg=built["fit_model_active_bridge"]["env_cfg"],
        num_features=int(fit_model_config["prior"]["num_features"]),
        device=device_obj,
    )
    validation_e2e_compare = _run_validation_e2e_compare(
        built=built,
        configs={
            "fit_model_active_bridge": fit_model_config,
            "phase3_anchor_reference": anchor_config,
        },
    )

    report = {
        "audit_entry": "fit_model_phase3_anchor_alignment_probe",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "rollout_seed": int(rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "fixed_env_contract": _fixed_env_contract_summary(frozen_h),
        "training_signature": {
            "fit_model_active_bridge": fit_model_sig,
            "phase3_anchor_reference": anchor_sig,
            "diff": sig_diff,
            "exact_match": len(sig_diff) == 0,
        },
        "fit_model_runtime": built["fit_model_active_bridge"]["runtime_summary"],
        "anchor_runtime": built["phase3_anchor_reference"]["runtime_summary"],
        "modes": {
            "fit_model_active_bridge": {
                **reports["fit_model_active_bridge"],
                "build_wall_s": float(built["fit_model_active_bridge"]["build_wall_s"]),
            },
            "phase3_anchor_reference": {
                **reports["phase3_anchor_reference"],
                "build_wall_s": float(built["phase3_anchor_reference"]["build_wall_s"]),
            },
        },
        "training_compare": {
            "grad_compare": {
                "cosine_similarity": float(cosine),
                "l2_delta_norm": float(torch.linalg.vector_norm(grad_delta).item()),
                "fit_model_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
                "anchor_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
            },
            "saved_policy_state_compare": saved_state_compare,
            "fit_model_vs_anchor_action_head": _tensor_compare_summary(
                fit_model_policy_state["action_net.weight"],
                anchor_policy_state["action_net.weight"],
            ),
            "fit_model_vs_anchor_value_head": _tensor_compare_summary(
                fit_model_policy_state["value_net.weight"],
                anchor_policy_state["value_net.weight"],
            ),
        },
        "validation_compare": validation_compare,
        "validation_e2e_compare": validation_e2e_compare,
    }
    report["localized_read"] = _localize(report)

    for mode_name in built:
        built[mode_name]["vec_env"].close()
        del built[mode_name]["algo"], built[mode_name]["callback"], built[mode_name]["vec_env"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gradient/output compare between the active fit_model rlpfn PPO bridge and the phase3 anchor contract."
    )
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rollout-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = run_fit_model_phase3_anchor_alignment_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        rollout_seed=args.rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    out = Path(args.output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
