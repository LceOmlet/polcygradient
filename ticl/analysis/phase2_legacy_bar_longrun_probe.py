import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _build_audit_env_cfg,
    _build_audit_ppo_policy_step_fn,
    _make_deterministic_batch_plan,
    _measure_current_policy_rollout_critic_quality,
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
    _run_eval,
    _seed_all,
)
from ticl.analysis.fixed_env_h import summarize_fixed_env_h
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated
from ticl.analysis.prior_generalization_audit import _build_zero_policy_step_fn
from ticl.config_utils import str2bool
from ticl.model_builder import get_model, load_model
from ticl.model_configs import get_model_default_config
from ticl.rlpfn_maintained_path import validate_rlpfn_maintained_path_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo
from stable_baselines3.common.logger import configure as configure_logger


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_full_frozen_h(path: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"full frozen_h JSON must decode to a dict, got {type(payload).__name__} from {resolved}"
        )
    return payload, resolved


def _sanitize_loaded_frozen_h_for_batch_use(frozen_h: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(frozen_h)
    # Once a specific frozen_h has been persisted and reloaded, exact-SCM rejection
    # knobs are no longer part of the environment definition. Keeping them active
    # would only re-trigger single-env-only guards in batch reset paths.
    if "reward_state_input_gain_fraction_rejection_min" in sanitized:
        sanitized["reward_state_input_gain_fraction_rejection_min"] = 0.0
    return sanitized


def _suite_from_seed_range(*, frozen_h, env_seed_start: int, rollout_seed_start: int, count: int) -> dict:
    env_seeds = list(range(int(env_seed_start), int(env_seed_start) + int(count)))
    rollout_seeds = list(range(int(rollout_seed_start), int(rollout_seed_start) + int(count)))
    return {
        "h_list": _clone_h_repeated(frozen_h, len(env_seeds)),
        "env_seeds": [int(v) for v in env_seeds],
        "rollout_seeds": [int(v) for v in rollout_seeds],
        "batch_size": len(env_seeds),
        "suite_seed": None,
    }


def _resolve_model_and_config(
    *,
    checkpoint_path: str | None,
    from_scratch: bool,
    from_scratch_model_type: str,
    device_obj: torch.device,
    build_seed: int,
):
    if bool(from_scratch):
        model_type = str(from_scratch_model_type).strip().lower()
        if model_type != "rlpfn":
            raise ValueError(
                f"phase2_legacy_bar_longrun_probe only supports from-scratch model_type='rlpfn', got {model_type!r}."
            )
        _seed_all(int(build_seed))
        config = copy.deepcopy(get_model_default_config(model_type))
        validate_rlpfn_maintained_path_config(config)
        _, model, _, _ = get_model(config, device=str(device_obj), should_train=False, verbose=False)
        model.to(device_obj)
        model.eval()
        return model, config

    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided unless from_scratch=True.")
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    return model, config


def _build_algo(
    *,
    checkpoint_path: str | None,
    device_obj: torch.device,
    frozen_h,
    env_cfg: dict,
    num_features: int,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
    build_seed: int,
    strict_native_rollout: bool,
    value_head_impl: str,
    value_path_adapter_impl: str,
    value_head_mlp_hidden_dim: int,
    space_contract: str,
    ppo_separate_value_backbone: bool,
    from_scratch: bool,
    from_scratch_model_type: str,
    vf_coef_override: float | None,
    policy_loss_coef_override: float | None,
    behavior_policy: str,
    deterministic_actor_sampling: bool,
    sb3_reward_normalization_enabled: bool,
    sb3_reward_normalization_clip: float,
    sb3_reward_normalization_epsilon: float,
):
    behavior_policy_resolved = str(behavior_policy).strip().lower()
    if behavior_policy_resolved not in {"learned", "unit_gaussian"}:
        raise ValueError(
            f"Unsupported behavior_policy={behavior_policy!r}. Expected 'learned' or 'unit_gaussian'."
        )
    _seed_all(int(build_seed))
    model, config = _resolve_model_and_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
    )
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    _apply_boundary_contract_mode_flags(prior, "normal")
    _apply_sep_state_reset_flag(prior, bool(ppo_reset_env_state_at_sep))

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(num_features),
        n_envs=1,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(config["optimizer"].get("ppo_gamma", 0.98)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.90)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        space_contract=str(space_contract),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_reward_normalization_clip=float(sb3_reward_normalization_clip),
        sb3_reward_normalization_epsilon=float(sb3_reward_normalization_epsilon),
        actor_baseline_mode=str(actor_baseline_mode),
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=bool(ppo_separate_value_backbone),
        value_head_impl=str(value_head_impl),
        value_path_adapter_impl=str(value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(value_head_mlp_hidden_dim),
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=(
            float(vf_coef_override)
            if vf_coef_override is not None
            else float(config["optimizer"].get("ppo_vf_coef", 0.5))
        ),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(train_env_seed)],
        rollout_rng_seeds=[int(train_rollout_seed)],
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
        deterministic_batch_plan=True,
        strict_native_rollout=bool(strict_native_rollout),
        restore_validation_policy_state=False,
        verbose=0,
    )
    algo._rwkv_policy_loss_coef = (
        float(policy_loss_coef_override)
        if policy_loss_coef_override is not None
        else 1.0
    )
    if behavior_policy_resolved == "unit_gaussian":
        _set_fixed_unit_gaussian_collect_policy(algo)
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo.set_logger(configure_logger(folder=None, format_strings=[]))
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    return algo, callback, vec_env


def _build_fixed_unit_gaussian_policy_step_fn(*, sampled: bool):
    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        del obs_t, reward_t, reward_mask_t, step_idx, env_info
        action_mean = torch.zeros_like(action_t[:, : int(action_t.shape[-1])], dtype=action_t.dtype)
        action_std = torch.ones_like(action_mean, dtype=action_mean.dtype)
        return (
            {
                "action_mean": action_mean,
                "action_std": action_std,
            },
            cache,
        )

    if bool(sampled):
        _policy_step_fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
            lambda actor_outputs, noise: noise.to(
                device=actor_outputs["action_mean"].device,
                dtype=actor_outputs["action_mean"].dtype,
            )
        )
    return _policy_step_fn


def _wrap_policy_step_fn_with_fixed_unit_gaussian(base_policy_step_fn):
    def _policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        policy_out = base_policy_step_fn(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            cache,
            step_idx,
            env_info,
        )
        cache_out = None
        main_out = policy_out
        if isinstance(policy_out, tuple):
            if len(policy_out) != 2:
                raise ValueError(
                    "Wrapped policy_step_fn must return output or (output, cache); "
                    f"got tuple len={len(policy_out)}"
                )
            main_out, cache_out = policy_out
        if not isinstance(main_out, dict):
            raise ValueError("Fixed unit-gaussian wrapper expects dict actor outputs from the PPO policy step fn.")
        if not torch.is_tensor(main_out.get("action_mean", None)):
            raise ValueError("Fixed unit-gaussian wrapper requires actor outputs to include action_mean.")
        fixed_out = dict(main_out)
        fixed_out["action_mean"] = torch.zeros_like(main_out["action_mean"], dtype=main_out["action_mean"].dtype)
        fixed_out["action_std"] = torch.ones_like(fixed_out["action_mean"], dtype=fixed_out["action_mean"].dtype)
        return (fixed_out, cache_out) if isinstance(policy_out, tuple) else fixed_out

    for attr_name in ("_clear_buffers", "_policy_actor_log_prob_fn", "_policy_actor_score_fn"):
        if hasattr(base_policy_step_fn, attr_name):
            setattr(_policy_step_fn, attr_name, getattr(base_policy_step_fn, attr_name))
    _policy_step_fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
        lambda actor_outputs, noise: noise.to(
            device=actor_outputs["action_mean"].device,
            dtype=actor_outputs["action_mean"].dtype,
        )
    )
    return _policy_step_fn


def _set_fixed_unit_gaussian_collect_policy(algo) -> None:
    original_make_step_fn = algo.policy.make_vectorized_rollout_step_fn
    algo._rwkv_deterministic_actor_sampling = False

    def _wrapped_make_step_fn():
        return _wrap_policy_step_fn_with_fixed_unit_gaussian(original_make_step_fn())

    algo.policy.make_vectorized_rollout_step_fn = _wrapped_make_step_fn  # type: ignore[method-assign]


def _collect_rollout(algo, callback, vec_env, *, progress_timestep: int, total_timesteps: int) -> None:
    algo._update_current_progress_remaining(int(progress_timestep), int(total_timesteps))
    ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if not ok:
        raise RuntimeError("collect_rollouts returned False during legacy-bar longrun probe.")


def _append_progress_row(progress_jsonl: Path | None, row: dict) -> None:
    if progress_jsonl is None:
        return
    progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl.open("a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")


_REWARD_COMPONENT_KEYS = (
    "ep_rew_mean",
    "ep_len_mean",
    "full_ep_rew_mean",
    "full_ep_len_mean",
    "reward_mean",
    "reward_std",
    "reward_return_mean",
    "reward_return_std",
    "reward_env_mean",
    "reward_env_std",
    "reward_env_return_mean",
    "reward_env_return_std",
    "reward_ctrl_mean",
    "reward_ctrl_std",
    "reward_ctrl_return_mean",
    "reward_ctrl_return_std",
    "reward_survival_mean",
    "reward_survival_std",
    "reward_survival_return_mean",
    "reward_survival_return_std",
    "reward_terminal_bonus_mean",
    "reward_terminal_bonus_std",
    "reward_terminal_bonus_return_mean",
    "reward_terminal_bonus_return_std",
)


def _extract_reward_component_summary(algo) -> dict[str, float | None]:
    raw_stats = dict(getattr(algo, "_rwkv_last_reward_component_stats", {}) or {})
    summary: dict[str, float | None] = {}
    for key in _REWARD_COMPONENT_KEYS:
        value = raw_stats.get(key, None)
        summary[str(key)] = None if value is None else float(value)
    return summary


def _format_optional_metric(value: float | None) -> str:
    return "na" if value is None else f"{float(value):+.6f}"


def _emit_epoch_summary_log(
    *,
    actor_baseline_mode: str,
    outer_epoch: int,
    outer_epochs: int,
    suite_smp_after_epoch: dict[str, Any],
    zero_full_mean: float,
    zero_suffix_mean: float,
    pre_rollout_reward_components: dict[str, float | None],
    next_rollout_reward_components: dict[str, float | None],
) -> None:
    print(
        "[phase2-longrun] "
        f"mode={str(actor_baseline_mode)} "
        f"epoch={int(outer_epoch)}/{int(outer_epochs)} "
        f"full_gap={float(suite_smp_after_epoch['full_return_mean'] - zero_full_mean):+.6f} "
        f"suffix_gap={float(suite_smp_after_epoch['suffix_return_mean'] - zero_suffix_mean):+.6f} "
        f"pre_env_ret={_format_optional_metric(pre_rollout_reward_components.get('reward_env_return_mean'))} "
        f"pre_ctrl_ret={_format_optional_metric(pre_rollout_reward_components.get('reward_ctrl_return_mean'))} "
        f"next_env_ret={_format_optional_metric(next_rollout_reward_components.get('reward_env_return_mean'))} "
        f"next_ctrl_ret={_format_optional_metric(next_rollout_reward_components.get('reward_ctrl_return_mean'))}"
    )


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
            "bundle_type": "phase2_legacy_bar_longrun_post_policy_bundle",
            "contract": copy.deepcopy(contract),
            "train_history": copy.deepcopy(train_history),
            "policy_state_dict": cpu_policy_state,
        },
        path,
    )
    return resolved


def _to_cpu_checkpoint_payload(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_checkpoint_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_checkpoint_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_checkpoint_payload(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


def _rng_state_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().detach().cpu().clone(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = [
            state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _running_mean_std_payload(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    required = ("mean", "var", "count")
    if not all(hasattr(obj, name) for name in required):
        return None
    return {
        "mean": np.asarray(getattr(obj, "mean"), dtype=np.float64).copy(),
        "var": np.asarray(getattr(obj, "var"), dtype=np.float64).copy(),
        "count": np.asarray(getattr(obj, "count"), dtype=np.float64).copy(),
    }


def _parse_float_csv(raw: str | None) -> tuple[float, ...]:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return ()
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _diagnostic_epoch_selected(
    *,
    epoch: int,
    start: int | None,
    end: int | None,
    every: int,
) -> bool:
    if int(every) <= 0 or start is None or end is None:
        return False
    epoch_i = int(epoch)
    start_i = int(start)
    end_i = int(end)
    return start_i <= epoch_i <= end_i and (epoch_i - start_i) % int(every) == 0


def _snapshot_path_for_epoch(
    *,
    snapshot_dir: Path | None,
    mode: str,
    epoch: int,
    suffix: str,
) -> Path | None:
    if snapshot_dir is None:
        return None
    return snapshot_dir / str(mode) / f"epoch_{int(epoch):05d}_{suffix}.json"


def _configure_train_update_snapshot(algo, path: Path | None) -> str | None:
    if path is None:
        algo._rwkv_train_outer_batch_snapshot_path = None
        algo._rwkv_train_outer_batch_snapshot_written = False
        algo._rwkv_last_train_outer_batch_snapshot_path = None
        return None
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    algo._rwkv_train_outer_batch_snapshot_path = str(resolved)
    algo._rwkv_train_outer_batch_snapshot_written = False
    algo._rwkv_last_train_outer_batch_snapshot_path = None
    algo._rwkv_train_outer_batch_snapshot_target_outer_batch_idx = 1
    algo._rwkv_train_outer_batch_snapshot_target_env_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_episode_index = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_start = None
    algo._rwkv_train_outer_batch_snapshot_target_objective_position_end = None
    return str(resolved)


def _configure_rollout_collect_snapshot(algo, path: Path | None) -> str | None:
    if path is None:
        algo._rwkv_rollout_collect_snapshot_path = None
        algo._rwkv_rollout_collect_snapshot_written = False
        algo._rwkv_last_rollout_collect_snapshot_path = None
        return None
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    algo._rwkv_rollout_collect_snapshot_path = str(resolved)
    algo._rwkv_rollout_collect_snapshot_written = False
    algo._rwkv_last_rollout_collect_snapshot_path = None
    return str(resolved)


def _rollout_buffer_checkpoint_payload(algo) -> dict[str, Any] | None:
    rollout_buffer = getattr(algo, "rollout_buffer", None)
    if rollout_buffer is None:
        return None
    keys = (
        "observations",
        "actions",
        "values",
        "log_probs",
        "advantages",
        "returns",
        "hidden_states_pi",
        "cell_states_pi",
        "hidden_states_vf",
        "cell_states_vf",
        "objective_masks",
        "rollout_return_means",
        "rollout_return_stds",
        "rollout_raw_returns",
        "actor_advantages",
        "value_target_bucket_idx",
        "episode_starts",
        "action_masks",
        "next_states",
        "next_state_masks",
    )
    payload: dict[str, Any] = {
        "buffer_size": int(getattr(rollout_buffer, "buffer_size", -1)),
        "n_envs": int(getattr(rollout_buffer, "n_envs", -1)),
        "generator_ready": bool(getattr(rollout_buffer, "generator_ready", False)),
        "next_state_dim": int(getattr(rollout_buffer, "next_state_dim", -1)),
    }
    for key in keys:
        if hasattr(rollout_buffer, key):
            payload[key] = _to_cpu_checkpoint_payload(getattr(rollout_buffer, key))
    for key in (
        "_flat_env_indices",
        "_flat_step_indices",
        "_flat_objective_episode_indices",
        "_flat_objective_episode_positions",
        "_flat_objective_global_positions",
    ):
        if hasattr(rollout_buffer, key):
            payload[key] = _to_cpu_checkpoint_payload(getattr(rollout_buffer, key))
    return payload


def _safe_reason_label(reasons: list[str]) -> str:
    if not reasons:
        return "manual"
    label = "_".join(str(reason).replace("/", "_").replace(" ", "_") for reason in reasons[:3])
    return label[:96] or "manual"


def _save_diagnostic_checkpoint(
    *,
    checkpoint_dir: Path,
    epoch: int,
    reasons: list[str],
    algo,
    contract: dict[str, Any],
    row: dict[str, Any],
    include_optimizer: bool,
    include_rollout_buffer: bool,
) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{int(epoch):05d}_{_safe_reason_label(reasons)}.pt"
    optimizer_state = None
    optimizer = getattr(getattr(algo, "policy", None), "optimizer", None)
    if bool(include_optimizer) and optimizer is not None:
        optimizer_state = _to_cpu_checkpoint_payload(optimizer.state_dict())
    payload = {
        "bundle_type": "phase2_legacy_bar_longrun_diagnostic_checkpoint",
        "epoch": int(epoch),
        "checkpoint_reasons": list(reasons),
        "contract": copy.deepcopy(contract),
        "row": copy.deepcopy(row),
        "policy_state_dict": {
            str(key): value.detach().cpu().clone()
            for key, value in algo.policy.state_dict().items()
        },
        "optimizer_state_dict": optimizer_state,
        "rng_state": _rng_state_payload(),
        "algo_state": {
            "num_timesteps": int(getattr(algo, "num_timesteps", 0)),
            "_n_updates": int(getattr(algo, "_n_updates", 0)),
            "_last_obs": _to_cpu_checkpoint_payload(getattr(algo, "_last_obs", None)),
            "_last_episode_starts": _to_cpu_checkpoint_payload(
                getattr(algo, "_last_episode_starts", None)
            ),
            "sb3_reward_normalization_enabled": bool(
                getattr(algo, "_rwkv_sb3_reward_normalization_enabled", False)
            ),
            "sb3_reward_normalization_clip": float(
                getattr(algo, "_rwkv_sb3_reward_normalization_clip", 10.0)
            ),
            "sb3_reward_normalization_epsilon": float(
                getattr(algo, "_rwkv_sb3_reward_normalization_epsilon", 1e-8)
            ),
            "sb3_reward_normalization_returns": _to_cpu_checkpoint_payload(
                getattr(algo, "_rwkv_sb3_reward_normalization_returns", None)
            ),
            "sb3_reward_normalization_ret_rms": _running_mean_std_payload(
                getattr(algo, "_rwkv_sb3_reward_normalization_ret_rms", None)
            ),
            "last_collect_rollout_env_rng_seeds": _to_cpu_checkpoint_payload(
                getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None)
            ),
            "last_collect_rollout_rollout_rng_seeds": _to_cpu_checkpoint_payload(
                getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None)
            ),
        },
    }
    if bool(include_rollout_buffer):
        payload["rollout_buffer"] = _rollout_buffer_checkpoint_payload(algo)
    torch.save(payload, path)
    return str(path.expanduser().resolve())


def _diagnostic_checkpoint_reasons(
    *,
    epoch: int,
    full_gap: float,
    best_full_gap_before_epoch: float | None,
    thresholds: tuple[float, ...],
    thresholds_seen: set[float],
    save_best: bool,
    save_every: int,
    dense_start: int | None,
    dense_end: int | None,
    dense_every: int,
) -> list[str]:
    reasons: list[str] = []
    if int(save_every) > 0 and int(epoch) % int(save_every) == 0:
        reasons.append(f"every{int(save_every)}")
    if (
        int(dense_every) > 0
        and dense_start is not None
        and dense_end is not None
        and int(dense_start) <= int(epoch) <= int(dense_end)
        and (int(epoch) - int(dense_start)) % int(dense_every) == 0
    ):
        reasons.append(f"dense{int(dense_every)}")
    if bool(save_best) and (
        best_full_gap_before_epoch is None or float(full_gap) > float(best_full_gap_before_epoch)
    ):
        reasons.append("best")
    if best_full_gap_before_epoch is not None:
        for threshold in thresholds:
            if threshold in thresholds_seen:
                continue
            if float(best_full_gap_before_epoch) >= float(threshold) and float(full_gap) <= float(threshold):
                thresholds_seen.add(threshold)
                reasons.append(f"first_below_{float(threshold):g}")
    return reasons


def _run_mode(
    *,
    checkpoint_path: str | None,
    device_obj: torch.device,
    env_cfg: dict,
    num_features: int,
    frozen_h,
    build_seed: int,
    actor_baseline_mode: str,
    train_env_seed: int,
    train_rollout_seed: int,
    suite: dict,
    zero_suffix_mean: float,
    zero_full_mean: float,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    outer_epochs: int,
    ppo_reset_env_state_at_sep: bool,
    strict_native_rollout: bool,
    progress_jsonl: Path | None,
    value_head_impl: str,
    value_path_adapter_impl: str,
    value_head_mlp_hidden_dim: int,
    space_contract: str,
    ppo_separate_value_backbone: bool,
    from_scratch: bool,
    from_scratch_model_type: str,
    vf_coef_override: float | None,
    policy_loss_coef_override: float | None,
    behavior_policy: str,
    deterministic_actor_sampling: bool,
    sb3_reward_normalization_enabled: bool,
    sb3_reward_normalization_clip: float,
    sb3_reward_normalization_epsilon: float,
    save_model_path: str | None,
    save_model_contract: dict[str, Any],
    diagnostic_checkpoint_dir: Path | None,
    diagnostic_checkpoint_save_every: int,
    diagnostic_checkpoint_dense_start: int | None,
    diagnostic_checkpoint_dense_end: int | None,
    diagnostic_checkpoint_dense_every: int,
    diagnostic_checkpoint_thresholds: tuple[float, ...],
    diagnostic_checkpoint_save_best: bool,
    diagnostic_checkpoint_include_optimizer: bool,
    diagnostic_checkpoint_include_rollout_buffer: bool,
    diagnostic_snapshot_dir: Path | None,
    diagnostic_snapshot_start: int | None,
    diagnostic_snapshot_end: int | None,
    diagnostic_snapshot_every: int,
) -> dict:
    algo, callback, vec_env = _build_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        env_cfg=env_cfg,
        num_features=int(num_features),
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(actor_baseline_mode),
        build_seed=int(build_seed),
        strict_native_rollout=bool(strict_native_rollout),
        value_head_impl=str(value_head_impl),
        value_path_adapter_impl=str(value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(value_head_mlp_hidden_dim),
        space_contract=str(space_contract),
        ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        vf_coef_override=vf_coef_override,
        policy_loss_coef_override=policy_loss_coef_override,
        behavior_policy=str(behavior_policy),
        deterministic_actor_sampling=bool(deterministic_actor_sampling),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_reward_normalization_clip=float(sb3_reward_normalization_clip),
        sb3_reward_normalization_epsilon=float(sb3_reward_normalization_epsilon),
    )
    total_timesteps = int(n_steps) * int(outer_epochs)
    history = []
    diagnostic_checkpoint_paths: list[str] = []
    diagnostic_thresholds_seen: set[float] = set()
    diagnostic_best_full_gap: float | None = None
    behavior_policy_resolved = str(behavior_policy).strip().lower()
    if behavior_policy_resolved == "unit_gaussian":
        sampled_policy_step_fn = _build_fixed_unit_gaussian_policy_step_fn(sampled=True)
        deterministic_policy_step_fn = _build_fixed_unit_gaussian_policy_step_fn(sampled=False)
    else:
        sampled_policy_step_fn = _build_audit_ppo_policy_step_fn(
            algo.policy,
            sampled=True,
            single_eval_pos=int(single_eval_pos),
            boundary_contract_mode="normal",
            deterministic_actor_sampling=True,
        )
        deterministic_policy_step_fn = _build_audit_ppo_policy_step_fn(
            algo.policy,
            sampled=False,
            single_eval_pos=int(single_eval_pos),
            boundary_contract_mode="normal",
            deterministic_actor_sampling=True,
        )
    try:
        pre_det = _run_eval(
            prior=algo._rwkv_env_prior,
            suite=suite,
            policy_step_fn=deterministic_policy_step_fn,
            n_samples=int(n_steps),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )
        pre_smp = _run_eval(
            prior=algo._rwkv_env_prior,
            suite=suite,
            policy_step_fn=sampled_policy_step_fn,
            n_samples=int(n_steps),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )
        initial_rollout_snapshot_path = _configure_rollout_collect_snapshot(
            algo,
            _snapshot_path_for_epoch(
                snapshot_dir=diagnostic_snapshot_dir
                if _diagnostic_epoch_selected(
                    epoch=1,
                    start=diagnostic_snapshot_start,
                    end=diagnostic_snapshot_end,
                    every=diagnostic_snapshot_every,
                )
                else None,
                mode=str(actor_baseline_mode),
                epoch=1,
                suffix="pre_update_rollout_collect_snapshot",
            ),
        )
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        initial_rollout_snapshot_written_path = (
            None
            if initial_rollout_snapshot_path is None
            else getattr(algo, "_rwkv_last_rollout_collect_snapshot_path", None)
        )
        initial_rollout_reward_components = _extract_reward_component_summary(algo)
        for outer_idx in range(int(outer_epochs)):
            epoch_i = int(outer_idx + 1)
            snapshot_selected = _diagnostic_epoch_selected(
                epoch=epoch_i,
                start=diagnostic_snapshot_start,
                end=diagnostic_snapshot_end,
                every=diagnostic_snapshot_every,
            )
            train_snapshot_path = _configure_train_update_snapshot(
                algo,
                _snapshot_path_for_epoch(
                    snapshot_dir=diagnostic_snapshot_dir if snapshot_selected else None,
                    mode=str(actor_baseline_mode),
                    epoch=epoch_i,
                    suffix="train_outer_batch_snapshot",
                ),
            )
            pre_quality = _measure_rollout_critic_quality(algo)
            pre_rollout_reward_components = _extract_reward_component_summary(algo)
            algo.train()
            train_snapshot_written_path = (
                None
                if train_snapshot_path is None
                else getattr(algo, "_rwkv_last_train_outer_batch_snapshot_path", None)
            )
            post_same_quality = _measure_current_policy_rollout_critic_quality(algo)
            suite_smp_after_epoch = _run_eval(
                prior=algo._rwkv_env_prior,
                suite=suite,
                policy_step_fn=sampled_policy_step_fn,
                n_samples=int(n_steps),
                num_features=int(num_features),
                single_eval_pos=int(single_eval_pos),
                device=device_obj,
            )
            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            next_epoch_i = int(epoch_i + 1)
            next_snapshot_selected = _diagnostic_epoch_selected(
                epoch=next_epoch_i,
                start=diagnostic_snapshot_start,
                end=diagnostic_snapshot_end,
                every=diagnostic_snapshot_every,
            )
            next_rollout_snapshot_path = _configure_rollout_collect_snapshot(
                algo,
                _snapshot_path_for_epoch(
                    snapshot_dir=diagnostic_snapshot_dir if next_snapshot_selected else None,
                    mode=str(actor_baseline_mode),
                    epoch=next_epoch_i,
                    suffix="pre_update_rollout_collect_snapshot",
                ),
            )
            _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
            next_rollout_snapshot_written_path = (
                None
                if next_rollout_snapshot_path is None
                else getattr(algo, "_rwkv_last_rollout_collect_snapshot_path", None)
            )
            walk_forward_quality = _measure_rollout_critic_quality(algo)
            next_rollout_reward_components = _extract_reward_component_summary(algo)
            row = {
                "outer_epoch": epoch_i,
                "actor_baseline_mode": str(actor_baseline_mode),
                "behavior_policy": str(behavior_policy_resolved),
                "pre_rollout": pre_quality,
                "pre_rollout_reward_components": pre_rollout_reward_components,
                "post_same_rollout": post_same_quality,
                "next_rollout": walk_forward_quality,
                "next_rollout_reward_components": next_rollout_reward_components,
                "suite_sampled_after_epoch": {
                    "full_return_mean": float(suite_smp_after_epoch["full_return_mean"]),
                    "suffix_return_mean": float(suite_smp_after_epoch["suffix_return_mean"]),
                    "full_gap_vs_zero": float(suite_smp_after_epoch["full_return_mean"] - zero_full_mean),
                    "suffix_gap_vs_zero": float(suite_smp_after_epoch["suffix_return_mean"] - zero_suffix_mean),
                },
            }
            diagnostic_snapshot_row: dict[str, Any] = {}
            if initial_rollout_snapshot_path is not None and epoch_i == 1:
                diagnostic_snapshot_row["initial_pre_update_rollout_collect_snapshot"] = {
                    "train_epoch": 1,
                    "path": initial_rollout_snapshot_path,
                    "written_path": initial_rollout_snapshot_written_path,
                    "written": bool(initial_rollout_snapshot_written_path),
                }
            if train_snapshot_path is not None:
                diagnostic_snapshot_row["train_outer_batch_snapshot"] = {
                    "train_epoch": epoch_i,
                    "path": train_snapshot_path,
                    "written_path": train_snapshot_written_path,
                    "written": bool(train_snapshot_written_path),
                }
            if next_rollout_snapshot_path is not None:
                diagnostic_snapshot_row["next_pre_update_rollout_collect_snapshot"] = {
                    "train_epoch": next_epoch_i,
                    "path": next_rollout_snapshot_path,
                    "written_path": next_rollout_snapshot_written_path,
                    "written": bool(next_rollout_snapshot_written_path),
                }
            if diagnostic_snapshot_row:
                row["diagnostic_snapshots"] = diagnostic_snapshot_row
            full_gap = float(row["suite_sampled_after_epoch"]["full_gap_vs_zero"])
            diagnostic_reasons = _diagnostic_checkpoint_reasons(
                epoch=int(outer_idx + 1),
                full_gap=full_gap,
                best_full_gap_before_epoch=diagnostic_best_full_gap,
                thresholds=diagnostic_checkpoint_thresholds,
                thresholds_seen=diagnostic_thresholds_seen,
                save_best=bool(diagnostic_checkpoint_save_best),
                save_every=int(diagnostic_checkpoint_save_every),
                dense_start=diagnostic_checkpoint_dense_start,
                dense_end=diagnostic_checkpoint_dense_end,
                dense_every=int(diagnostic_checkpoint_dense_every),
            )
            if diagnostic_best_full_gap is None or full_gap > float(diagnostic_best_full_gap):
                diagnostic_best_full_gap = float(full_gap)
            if diagnostic_checkpoint_dir is not None and diagnostic_reasons:
                saved_checkpoint = _save_diagnostic_checkpoint(
                    checkpoint_dir=diagnostic_checkpoint_dir,
                    epoch=int(outer_idx + 1),
                    reasons=diagnostic_reasons,
                    algo=algo,
                    contract={
                        **save_model_contract,
                        "actor_baseline_mode": str(actor_baseline_mode),
                        "diagnostic_checkpoint": {
                            "save_every": int(diagnostic_checkpoint_save_every),
                            "dense_start": diagnostic_checkpoint_dense_start,
                            "dense_end": diagnostic_checkpoint_dense_end,
                            "dense_every": int(diagnostic_checkpoint_dense_every),
                            "thresholds": list(diagnostic_checkpoint_thresholds),
                            "save_best": bool(diagnostic_checkpoint_save_best),
                            "include_optimizer": bool(diagnostic_checkpoint_include_optimizer),
                            "include_rollout_buffer": bool(diagnostic_checkpoint_include_rollout_buffer),
                        },
                    },
                    row=row,
                    include_optimizer=bool(diagnostic_checkpoint_include_optimizer),
                    include_rollout_buffer=bool(diagnostic_checkpoint_include_rollout_buffer),
                )
                diagnostic_checkpoint_paths.append(saved_checkpoint)
                row["diagnostic_checkpoint"] = {
                    "saved": True,
                    "path": saved_checkpoint,
                    "reasons": list(diagnostic_reasons),
                }
                print(
                    "[phase2-longrun] "
                    f"diagnostic_checkpoint epoch={int(outer_idx + 1)} "
                    f"reasons={','.join(diagnostic_reasons)} "
                    f"path={saved_checkpoint}"
                )
            elif diagnostic_reasons:
                row["diagnostic_checkpoint"] = {
                    "saved": False,
                    "path": None,
                    "reasons": list(diagnostic_reasons),
                }
            history.append(row)
            _append_progress_row(
                progress_jsonl,
                {
                    "mode": str(actor_baseline_mode),
                    **row,
                },
            )
            _emit_epoch_summary_log(
                actor_baseline_mode=str(actor_baseline_mode),
                outer_epoch=int(outer_idx + 1),
                outer_epochs=int(outer_epochs),
                suite_smp_after_epoch=suite_smp_after_epoch,
                zero_full_mean=float(zero_full_mean),
                zero_suffix_mean=float(zero_suffix_mean),
                pre_rollout_reward_components=pre_rollout_reward_components,
                next_rollout_reward_components=next_rollout_reward_components,
            )
        final_rollout_reward_components = _extract_reward_component_summary(algo)
        post_det = _run_eval(
            prior=algo._rwkv_env_prior,
            suite=suite,
            policy_step_fn=deterministic_policy_step_fn,
            n_samples=int(n_steps),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )
        post_smp = _run_eval(
            prior=algo._rwkv_env_prior,
            suite=suite,
            policy_step_fn=sampled_policy_step_fn,
            n_samples=int(n_steps),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device=device_obj,
        )
        saved_model_path = None
        if save_model_path is not None:
            saved_model_path = _save_post_policy_bundle(
                bundle_path=str(save_model_path),
                algo=algo,
                contract=save_model_contract,
                train_history={
                    "history": history,
                    "final_rollout_reward_components": final_rollout_reward_components,
                },
            )
            print(f"[phase2-longrun] saved_model_path={saved_model_path}")
        return {
            "effective_actor_gae_space": str(getattr(algo, "_rwkv_actor_gae_space", "normalized")),
            "space_contract": getattr(algo, "_rwkv_space_contract", None),
            "behavior_policy": str(behavior_policy_resolved),
            "initial_rollout_reward_components": initial_rollout_reward_components,
            "final_rollout_reward_components": final_rollout_reward_components,
            "post_policy_bundle": {
                "saved": bool(saved_model_path is not None),
                "save_path": saved_model_path,
            },
            "diagnostic_checkpoints": {
                "count": int(len(diagnostic_checkpoint_paths)),
                "paths": list(diagnostic_checkpoint_paths),
            },
            "pre_det_gap": float(pre_det["suffix_return_mean"] - zero_suffix_mean),
            "post_det_gap": float(post_det["suffix_return_mean"] - zero_suffix_mean),
            "delta_det_gap": float(post_det["suffix_return_mean"] - pre_det["suffix_return_mean"]),
            "pre_smp_gap": float(pre_smp["suffix_return_mean"] - zero_suffix_mean),
            "post_smp_gap": float(post_smp["suffix_return_mean"] - zero_suffix_mean),
            "delta_smp_gap": float(post_smp["suffix_return_mean"] - pre_smp["suffix_return_mean"]),
            "pre_full_smp_gap": float(pre_smp["full_return_mean"] - zero_full_mean),
            "post_full_smp_gap": float(post_smp["full_return_mean"] - zero_full_mean),
            "delta_full_smp_gap": float(post_smp["full_return_mean"] - pre_smp["full_return_mean"]),
            "final_reward_env_return_mean": final_rollout_reward_components.get("reward_env_return_mean"),
            "final_reward_ctrl_return_mean": final_rollout_reward_components.get("reward_ctrl_return_mean"),
            "history": history,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_phase2_legacy_bar_longrun_probe(
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
    from_scratch: bool = False,
    from_scratch_model_type: str = "rlpfn",
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    eval_env_seed_start: int = 3000,
    eval_rollout_seed_start: int = 4000,
    eval_env_count: int = 4,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    outer_epochs: int = 8,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    strict_native_rollout: bool = False,
    value_head_impl: str = "legacy_bar",
    value_path_adapter_impl: str = "none",
    value_head_mlp_hidden_dim: int = 256,
    space_contract: str = "normalized",
    ppo_separate_value_backbone: bool = False,
    vf_coef_override: float | None = None,
    policy_loss_coef_override: float | None = None,
    behavior_policy: str = "learned",
    deterministic_actor_sampling: bool = True,
    sb3_reward_normalization_enabled: bool = False,
    sb3_reward_normalization_clip: float = 10.0,
    sb3_reward_normalization_epsilon: float = 1e-8,
    modes: list[str] | None = None,
    progress_jsonl: str | None = None,
    fixed_frozen_h_json: str | None = None,
    save_model_path: str | None = None,
    diagnostic_checkpoint_dir: str | None = None,
    diagnostic_checkpoint_save_every: int = 0,
    diagnostic_checkpoint_dense_start: int | None = None,
    diagnostic_checkpoint_dense_end: int | None = None,
    diagnostic_checkpoint_dense_every: int = 0,
    diagnostic_checkpoint_thresholds: str | None = None,
    diagnostic_checkpoint_save_best: bool = False,
    diagnostic_checkpoint_include_optimizer: bool = True,
    diagnostic_checkpoint_include_rollout_buffer: bool = False,
    diagnostic_snapshot_dir: str | None = None,
    diagnostic_snapshot_start: int | None = None,
    diagnostic_snapshot_end: int | None = None,
    diagnostic_snapshot_every: int = 0,
) -> dict:
    if int(n_steps) <= int(single_eval_pos):
        raise ValueError(
            f"n_steps must exceed single_eval_pos for objective tokens to exist, got n_steps={int(n_steps)} "
            f"and single_eval_pos={int(single_eval_pos)}."
        )
    if bool(from_scratch) and checkpoint_path is not None:
        raise ValueError("Provide either checkpoint_path or from_scratch=True, not both.")
    if (not bool(from_scratch)) and checkpoint_path is None:
        raise ValueError("checkpoint_path is required unless from_scratch=True.")
    device_obj = torch.device(str(device or _default_device()))
    _, config = _resolve_model_and_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    num_features = int(config["prior"]["num_features"])
    loaded_frozen_h_path = None
    if fixed_frozen_h_json is None:
        fixed_bundle = _prepare_audit_fixed_env_contract(
            env_cfg=env_cfg,
            boundary_contract_mode="normal",
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            frozen_h_seed=int(frozen_h_seed),
            frozen_h_seed_mode="current",
        )
    else:
        frozen_h_loaded, loaded_frozen_h_path = _load_full_frozen_h(str(fixed_frozen_h_json))
        frozen_h_loaded = _sanitize_loaded_frozen_h_for_batch_use(frozen_h_loaded)
        prior_for_h = EnvironmentPrior(copy.deepcopy(env_cfg))
        _apply_boundary_contract_mode_flags(prior_for_h, "normal")
        _apply_sep_state_reset_flag(prior_for_h, bool(ppo_reset_env_state_at_sep))
        fixed_bundle = {
            "prior_for_h": prior_for_h,
            "frozen_h": frozen_h_loaded,
            "fixed_env_contract": {
                "source_full_frozen_h_json": str(loaded_frozen_h_path),
                "frozen_h_seed": int(frozen_h_seed),
                "frozen_h_seed_mode": "loaded_from_json",
                "zero_eval_prior_reuses_sampling_state": False,
                **summarize_fixed_env_h(frozen_h_loaded),
            },
        }
    frozen_h = fixed_bundle["frozen_h"]
    suite = _suite_from_seed_range(
        frozen_h=frozen_h,
        env_seed_start=int(eval_env_seed_start),
        rollout_seed_start=int(eval_rollout_seed_start),
        count=int(eval_env_count),
    )
    prior_for_zero = fixed_bundle["prior_for_h"]
    zero_metrics = _run_eval(
        prior=prior_for_zero,
        suite=suite,
        policy_step_fn=_build_zero_policy_step_fn(),
        n_samples=int(n_steps),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
    )
    zero_suffix_mean = float(zero_metrics["suffix_return_mean"])
    zero_full_mean = float(zero_metrics["full_return_mean"])
    progress_path = None if progress_jsonl is None else Path(progress_jsonl).expanduser().resolve()
    if progress_path is not None and progress_path.exists():
        progress_path.unlink()
    diagnostic_checkpoint_path = (
        None if diagnostic_checkpoint_dir is None else Path(diagnostic_checkpoint_dir).expanduser().resolve()
    )
    diagnostic_snapshot_path = (
        None if diagnostic_snapshot_dir is None else Path(diagnostic_snapshot_dir).expanduser().resolve()
    )
    diagnostic_threshold_values = _parse_float_csv(diagnostic_checkpoint_thresholds)

    report = {
        "audit_entry": "phase2_legacy_bar_longrun_probe",
        "init_mode": "from_scratch" if bool(from_scratch) else "checkpoint",
        "from_scratch_model_type": str(from_scratch_model_type),
        "checkpoint_path": None
        if checkpoint_path is None
        else str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "fixed_frozen_h_json": None if loaded_frozen_h_path is None else str(loaded_frozen_h_path),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "eval_env_seed_start": int(eval_env_seed_start),
        "eval_rollout_seed_start": int(eval_rollout_seed_start),
        "eval_env_count": int(eval_env_count),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "outer_epochs": int(outer_epochs),
        "build_seed": int(build_seed),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "strict_native_rollout": bool(strict_native_rollout),
        "ppo_separate_value_backbone": bool(ppo_separate_value_backbone),
        "value_head_impl": str(value_head_impl),
        "value_path_adapter_impl": str(value_path_adapter_impl),
        "value_head_mlp_hidden_dim": int(value_head_mlp_hidden_dim),
        "space_contract": str(space_contract),
        "vf_coef_override": None if vf_coef_override is None else float(vf_coef_override),
        "policy_loss_coef_override": (
            None if policy_loss_coef_override is None else float(policy_loss_coef_override)
        ),
        "deterministic_actor_sampling_collect": bool(deterministic_actor_sampling),
        "save_model_path": None if save_model_path is None else str(Path(save_model_path).expanduser().resolve()),
        "diagnostic_checkpoint_dir": (
            None if diagnostic_checkpoint_path is None else str(diagnostic_checkpoint_path)
        ),
        "diagnostic_checkpoint_save_every": int(diagnostic_checkpoint_save_every),
        "diagnostic_checkpoint_dense_start": (
            None if diagnostic_checkpoint_dense_start is None else int(diagnostic_checkpoint_dense_start)
        ),
        "diagnostic_checkpoint_dense_end": (
            None if diagnostic_checkpoint_dense_end is None else int(diagnostic_checkpoint_dense_end)
        ),
        "diagnostic_checkpoint_dense_every": int(diagnostic_checkpoint_dense_every),
        "diagnostic_checkpoint_thresholds": list(diagnostic_threshold_values),
        "diagnostic_checkpoint_save_best": bool(diagnostic_checkpoint_save_best),
        "diagnostic_checkpoint_include_optimizer": bool(diagnostic_checkpoint_include_optimizer),
        "diagnostic_checkpoint_include_rollout_buffer": bool(diagnostic_checkpoint_include_rollout_buffer),
        "diagnostic_snapshot_dir": (
            None if diagnostic_snapshot_path is None else str(diagnostic_snapshot_path)
        ),
        "diagnostic_snapshot_start": (
            None if diagnostic_snapshot_start is None else int(diagnostic_snapshot_start)
        ),
        "diagnostic_snapshot_end": (
            None if diagnostic_snapshot_end is None else int(diagnostic_snapshot_end)
        ),
        "diagnostic_snapshot_every": int(diagnostic_snapshot_every),
        "behavior_policy": str(behavior_policy),
        "modes": {},
    }
    save_model_contract = {
        "audit_entry": "phase2_legacy_bar_longrun_probe",
        "checkpoint_path": None
        if checkpoint_path is None
        else str(Path(checkpoint_path).expanduser().resolve()),
        "from_scratch": bool(from_scratch),
        "from_scratch_model_type": str(from_scratch_model_type),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "fixed_frozen_h_json": None if loaded_frozen_h_path is None else str(loaded_frozen_h_path),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "outer_epochs": int(outer_epochs),
        "ppo_separate_value_backbone": bool(ppo_separate_value_backbone),
        "space_contract": str(space_contract),
        "behavior_policy": str(behavior_policy),
        "deterministic_actor_sampling_collect": bool(deterministic_actor_sampling),
        "sb3_reward_normalization_enabled": bool(sb3_reward_normalization_enabled),
        "sb3_reward_normalization_clip": float(sb3_reward_normalization_clip),
        "sb3_reward_normalization_epsilon": float(sb3_reward_normalization_epsilon),
        "diagnostic_checkpoint_include_optimizer": bool(diagnostic_checkpoint_include_optimizer),
        "diagnostic_checkpoint_include_rollout_buffer": bool(diagnostic_checkpoint_include_rollout_buffer),
    }
    for mode in list(modes or ["learned", "zero"]):
        report["modes"][str(mode)] = _run_mode(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            env_cfg=env_cfg,
            num_features=int(num_features),
            frozen_h=frozen_h,
            build_seed=int(build_seed),
            actor_baseline_mode=str(mode),
            train_env_seed=int(train_env_seed),
            train_rollout_seed=int(train_rollout_seed),
            suite=suite,
            zero_suffix_mean=float(zero_suffix_mean),
            zero_full_mean=float(zero_full_mean),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            outer_epochs=int(outer_epochs),
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            strict_native_rollout=bool(strict_native_rollout),
            progress_jsonl=progress_path,
            value_head_impl=str(value_head_impl),
            value_path_adapter_impl=str(value_path_adapter_impl),
            value_head_mlp_hidden_dim=int(value_head_mlp_hidden_dim),
            space_contract=str(space_contract),
            ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
            from_scratch=bool(from_scratch),
            from_scratch_model_type=str(from_scratch_model_type),
            vf_coef_override=vf_coef_override,
            policy_loss_coef_override=policy_loss_coef_override,
            behavior_policy=str(behavior_policy),
            deterministic_actor_sampling=bool(deterministic_actor_sampling),
            sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
            sb3_reward_normalization_clip=float(sb3_reward_normalization_clip),
            sb3_reward_normalization_epsilon=float(sb3_reward_normalization_epsilon),
            save_model_path=save_model_path if str(mode) == "learned" else None,
            save_model_contract={**save_model_contract, "actor_baseline_mode": str(mode)},
            diagnostic_checkpoint_dir=(
                diagnostic_checkpoint_path / str(mode)
                if diagnostic_checkpoint_path is not None and str(mode) == "learned"
                else None
            ),
            diagnostic_checkpoint_save_every=int(diagnostic_checkpoint_save_every),
            diagnostic_checkpoint_dense_start=diagnostic_checkpoint_dense_start,
            diagnostic_checkpoint_dense_end=diagnostic_checkpoint_dense_end,
            diagnostic_checkpoint_dense_every=int(diagnostic_checkpoint_dense_every),
            diagnostic_checkpoint_thresholds=diagnostic_threshold_values,
            diagnostic_checkpoint_save_best=bool(diagnostic_checkpoint_save_best),
            diagnostic_checkpoint_include_optimizer=bool(diagnostic_checkpoint_include_optimizer),
            diagnostic_checkpoint_include_rollout_buffer=bool(diagnostic_checkpoint_include_rollout_buffer),
            diagnostic_snapshot_dir=(
                diagnostic_snapshot_path
                if diagnostic_snapshot_path is not None and str(mode) == "learned"
                else None
            ),
            diagnostic_snapshot_start=diagnostic_snapshot_start,
            diagnostic_snapshot_end=diagnostic_snapshot_end,
            diagnostic_snapshot_every=int(diagnostic_snapshot_every),
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2 legacy-bar longrun probe aligned with trusted short-run contracts."
    )
    parser.add_argument("checkpoint_path", nargs="?", default=None, type=str)
    parser.add_argument("--from-scratch", type=str2bool, default=False)
    parser.add_argument("--from-scratch-model-type", type=str, default="rlpfn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--eval-env-seed-start", type=int, default=3000)
    parser.add_argument("--eval-rollout-seed-start", type=int, default=4000)
    parser.add_argument("--eval-env-count", type=int, default=4)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=8)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", type=str2bool, default=True)
    parser.add_argument("--strict-native-rollout", type=str2bool, default=False)
    parser.add_argument("--value-head-impl", type=str, default="legacy_bar")
    parser.add_argument("--value-path-adapter-impl", type=str, default="none")
    parser.add_argument("--value-head-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--space-contract", type=str, default="normalized")
    parser.add_argument("--ppo-separate-value-backbone", type=str2bool, default=False)
    parser.add_argument("--vf-coef-override", type=float, default=None)
    parser.add_argument("--policy-loss-coef-override", type=float, default=None)
    parser.add_argument("--behavior-policy", type=str, default="learned")
    parser.add_argument("--deterministic-actor-sampling", type=str2bool, default=True)
    parser.add_argument("--sb3-reward-normalization-enabled", type=str2bool, default=False)
    parser.add_argument("--sb3-reward-normalization-clip", type=float, default=10.0)
    parser.add_argument("--sb3-reward-normalization-epsilon", type=float, default=1e-8)
    parser.add_argument("--modes", nargs="+", default=["learned", "zero"])
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_legacy_bar_longrun_probe.json",
    )
    parser.add_argument("--progress-jsonl", type=str, default=None)
    parser.add_argument("--fixed-frozen-h-json", type=str, default=None)
    parser.add_argument("--save-model-path", type=str, default=None)
    parser.add_argument("--diagnostic-checkpoint-dir", type=str, default=None)
    parser.add_argument("--diagnostic-checkpoint-save-every", type=int, default=0)
    parser.add_argument("--diagnostic-checkpoint-dense-start", type=int, default=None)
    parser.add_argument("--diagnostic-checkpoint-dense-end", type=int, default=None)
    parser.add_argument("--diagnostic-checkpoint-dense-every", type=int, default=0)
    parser.add_argument("--diagnostic-checkpoint-thresholds", type=str, default=None)
    parser.add_argument("--diagnostic-checkpoint-save-best", type=str2bool, default=False)
    parser.add_argument("--diagnostic-checkpoint-include-optimizer", type=str2bool, default=True)
    parser.add_argument("--diagnostic-checkpoint-include-rollout-buffer", type=str2bool, default=False)
    parser.add_argument("--diagnostic-snapshot-dir", type=str, default=None)
    parser.add_argument("--diagnostic-snapshot-start", type=int, default=None)
    parser.add_argument("--diagnostic-snapshot-end", type=int, default=None)
    parser.add_argument("--diagnostic-snapshot-every", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl = (
        args.progress_jsonl
        if args.progress_jsonl is not None
        else str(output_path.with_suffix(".progress.jsonl"))
    )
    report = run_phase2_legacy_bar_longrun_probe(
        checkpoint_path=args.checkpoint_path,
        from_scratch=bool(args.from_scratch),
        from_scratch_model_type=str(args.from_scratch_model_type),
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        eval_env_seed_start=int(args.eval_env_seed_start),
        eval_rollout_seed_start=int(args.eval_rollout_seed_start),
        eval_env_count=int(args.eval_env_count),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
        value_head_mlp_hidden_dim=int(args.value_head_mlp_hidden_dim),
        space_contract=str(args.space_contract),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        vf_coef_override=args.vf_coef_override,
        policy_loss_coef_override=args.policy_loss_coef_override,
        behavior_policy=str(args.behavior_policy),
        deterministic_actor_sampling=bool(args.deterministic_actor_sampling),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
        sb3_reward_normalization_clip=float(args.sb3_reward_normalization_clip),
        sb3_reward_normalization_epsilon=float(args.sb3_reward_normalization_epsilon),
        modes=list(args.modes),
        progress_jsonl=progress_jsonl,
        fixed_frozen_h_json=args.fixed_frozen_h_json,
        save_model_path=args.save_model_path,
        diagnostic_checkpoint_dir=args.diagnostic_checkpoint_dir,
        diagnostic_checkpoint_save_every=int(args.diagnostic_checkpoint_save_every),
        diagnostic_checkpoint_dense_start=args.diagnostic_checkpoint_dense_start,
        diagnostic_checkpoint_dense_end=args.diagnostic_checkpoint_dense_end,
        diagnostic_checkpoint_dense_every=int(args.diagnostic_checkpoint_dense_every),
        diagnostic_checkpoint_thresholds=args.diagnostic_checkpoint_thresholds,
        diagnostic_checkpoint_save_best=bool(args.diagnostic_checkpoint_save_best),
        diagnostic_checkpoint_include_optimizer=bool(args.diagnostic_checkpoint_include_optimizer),
        diagnostic_checkpoint_include_rollout_buffer=bool(args.diagnostic_checkpoint_include_rollout_buffer),
        diagnostic_snapshot_dir=args.diagnostic_snapshot_dir,
        diagnostic_snapshot_start=args.diagnostic_snapshot_start,
        diagnostic_snapshot_end=args.diagnostic_snapshot_end,
        diagnostic_snapshot_every=int(args.diagnostic_snapshot_every),
    )
    payload = json.dumps(report, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
