import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

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


def _single_env_suite(suite: dict, env_index: int) -> dict:
    idx = int(env_index)
    return {
        "h_list": [copy.deepcopy(list(suite["h_list"])[idx])],
        "env_seeds": [int(list(suite["env_seeds"])[idx])],
        "rollout_seeds": [int(list(suite["rollout_seeds"])[idx])],
        "batch_size": 1,
        "suite_seed": suite.get("suite_seed", None),
    }


def _record(bucket: dict[str, dict[str, float]], name: str, elapsed_s: float) -> None:
    rec = bucket.setdefault(name, {"calls": 0, "total_s": 0.0})
    rec["calls"] += 1
    rec["total_s"] += float(elapsed_s)


@contextmanager
def _replace_method(obj, method_name: str, wrapped):
    original = getattr(obj, method_name)
    setattr(obj, method_name, wrapped)
    try:
        yield original
    finally:
        setattr(obj, method_name, original)


def _build_instrumented_obs_fn(policy, bucket):
    def _instrumented(
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        reward_t: torch.Tensor,
        reward_mask_t: torch.Tensor,
        env_info: dict,
    ) -> torch.Tensor:
        t_total = time.perf_counter()

        def _vector_from_env_info(key: str, *, dtype: torch.dtype) -> torch.Tensor:
            value = env_info[key]
            if torch.is_tensor(value):
                out = value.to(device=obs_t.device, dtype=dtype)
            else:
                out = torch.full((batch_size,), value, device=obs_t.device, dtype=dtype)
            return out

        t0 = time.perf_counter()
        batch_size = int(obs_t.shape[0])
        obs = torch.zeros((batch_size, policy.num_features), device=obs_t.device, dtype=obs_t.dtype)
        obs_slot_dims = _vector_from_env_info("obs_slot_dim", dtype=torch.long)
        action_slot_dims = _vector_from_env_info("action_slot_dim", dtype=torch.long)
        action_dims = _vector_from_env_info("action_dim_per_sample", dtype=torch.long)
        terminal_enabled = _vector_from_env_info("terminal_reset_enabled", dtype=torch.bool)
        reward_idx = obs_slot_dims
        mask_idx = obs_slot_dims + 1
        phase_idx = obs_slot_dims + 2
        terminal_idx = obs_slot_dims + 3
        action_start = obs_slot_dims + 3 + terminal_enabled.to(dtype=torch.long)
        rows = torch.arange(batch_size, device=obs_t.device, dtype=torch.long)
        obs_cap = torch.minimum(
            _vector_from_env_info("obs_dim", dtype=torch.long),
            obs_slot_dims,
        )
        _record(bucket, "build_obs.env_info_and_index_setup", time.perf_counter() - t0)

        t0 = time.perf_counter()
        obs_write_cap = int(min(int(obs_t.shape[-1]), policy.num_features))
        if obs_write_cap > 0:
            cols = torch.arange(obs_write_cap, device=obs_t.device, dtype=torch.long).unsqueeze(0)
            valid = cols < obs_cap.unsqueeze(1)
            obs[:, :obs_write_cap] = obs_t[:, :obs_write_cap] * valid.to(dtype=obs.dtype)
        _record(bucket, "build_obs.obs_prefix_copy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        valid = reward_idx < policy.num_features
        if bool(valid.any().item()):
            obs[rows[valid], reward_idx[valid]] = reward_t.reshape(batch_size)[valid].to(dtype=obs.dtype)
        valid = mask_idx < policy.num_features
        if bool(valid.any().item()):
            obs[rows[valid], mask_idx[valid]] = reward_mask_t.reshape(batch_size)[valid].to(dtype=obs.dtype)
        phase_t = env_info.get("phase_t", None)
        if phase_t is not None:
            phase_scalar = phase_t.reshape(batch_size).to(device=obs.device, dtype=obs.dtype)
            valid = phase_idx < policy.num_features
            if bool(valid.any().item()):
                obs[rows[valid], phase_idx[valid]] = phase_scalar[valid]
        terminal_t = env_info.get("terminal_t", None)
        if terminal_t is not None:
            terminal_scalar = terminal_t.reshape(batch_size).to(device=obs.device, dtype=obs.dtype)
            valid = terminal_enabled & (terminal_idx < policy.num_features)
            if bool(valid.any().item()):
                obs[rows[valid], terminal_idx[valid]] = terminal_scalar[valid]
        _record(bucket, "build_obs.reward_mask_phase_terminal_write", time.perf_counter() - t0)

        t0 = time.perf_counter()
        action_cap = torch.minimum(action_dims, action_slot_dims)
        action_write_cap = int(min(int(action_t.shape[-1]), policy.num_features))
        if action_write_cap > 0:
            pos = torch.arange(action_write_cap, device=obs.device, dtype=torch.long).unsqueeze(0)
            dst_cols = action_start.unsqueeze(1) + pos
            valid = (pos < action_cap.unsqueeze(1)) & (dst_cols < policy.num_features)
            if bool(valid.any().item()):
                obs[rows.unsqueeze(1).expand_as(dst_cols)[valid], dst_cols[valid]] = action_t[:, :action_write_cap][valid]
        _record(bucket, "build_obs.action_tail_write", time.perf_counter() - t0)
        _record(bucket, "build_obs.total", time.perf_counter() - t_total)
        return obs

    return _instrumented


def _build_instrumented_terminal_tail_fn(prior, bucket):
    def _instrumented(
        terminal_signal,
        *,
        enabled,
        reset_prob,
        terminal_draw=None,
        signal_history=None,
        history_length=0,
    ):
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        if terminal_signal is None:
            raise ValueError("terminal_signal must be provided for tail-triggered reset")
        signal_t = terminal_signal
        if not torch.is_tensor(signal_t):
            signal_t = torch.as_tensor(signal_t, dtype=torch.float32)
        if signal_t.ndim > 0 and signal_t.shape[-1] == 1:
            signal_t = signal_t.squeeze(-1)
        enabled_t = enabled
        if not torch.is_tensor(enabled_t):
            enabled_t = torch.as_tensor(enabled_t, device=signal_t.device, dtype=torch.bool)
        else:
            enabled_t = enabled_t.to(device=signal_t.device, dtype=torch.bool)
        if enabled_t.ndim < signal_t.ndim:
            enabled_t = enabled_t.reshape(signal_t.shape)
        if not bool(enabled_t.any().item()):
            _record(bucket, "terminal_tail.setup", time.perf_counter() - t0)
            _record(bucket, "terminal_tail.total", time.perf_counter() - t_total)
            return torch.zeros_like(signal_t, dtype=torch.bool)
        prob_t = torch.as_tensor(reset_prob, device=signal_t.device, dtype=signal_t.dtype)
        while prob_t.ndim < signal_t.ndim:
            prob_t = prob_t.unsqueeze(-1)
        prob_t = torch.clamp(prob_t, min=0.0, max=1.0)
        if not bool((prob_t[enabled_t] > 0).any().item()):
            _record(bucket, "terminal_tail.setup", time.perf_counter() - t0)
            _record(bucket, "terminal_tail.total", time.perf_counter() - t_total)
            return torch.zeros_like(signal_t, dtype=torch.bool)
        if bool((prob_t[enabled_t] >= 1).all().item()):
            _record(bucket, "terminal_tail.setup", time.perf_counter() - t0)
            _record(bucket, "terminal_tail.total", time.perf_counter() - t_total)
            return enabled_t.clone()
        history_len = int(max(0, int(history_length)))
        enabled_idx = torch.nonzero(enabled_t.reshape(-1), as_tuple=False).reshape(-1)
        if enabled_idx.numel() <= 0:
            _record(bucket, "terminal_tail.setup", time.perf_counter() - t0)
            _record(bucket, "terminal_tail.total", time.perf_counter() - t_total)
            return torch.zeros_like(signal_t, dtype=torch.bool)
        draw_t = terminal_draw
        if draw_t is None:
            draw_t = torch.zeros_like(signal_t.reshape(-1))
        elif not torch.is_tensor(draw_t):
            draw_t = torch.as_tensor(draw_t, device=signal_t.device, dtype=signal_t.dtype)
        else:
            draw_t = draw_t.to(device=signal_t.device, dtype=signal_t.dtype)
        if draw_t.ndim > 0 and draw_t.shape[-1] == 1:
            draw_t = draw_t.squeeze(-1)
        draw_enabled = draw_t.reshape(-1).index_select(0, enabled_idx)
        prob_enabled = prob_t.reshape(-1).index_select(0, enabled_idx)
        signal_enabled = signal_t.reshape(-1).index_select(0, enabled_idx)
        terminal_enabled = torch.zeros_like(signal_enabled, dtype=torch.bool)
        _record(bucket, "terminal_tail.setup", time.perf_counter() - t0)

        history_t = signal_history
        if (history_t is not None) and (history_len > 0):
            t0 = time.perf_counter()
            if not torch.is_tensor(history_t):
                history_t = torch.as_tensor(history_t, device=signal_t.device, dtype=signal_t.dtype)
            else:
                history_t = history_t.to(device=signal_t.device, dtype=signal_t.dtype)
            if history_t.ndim == 1:
                history_t = history_t.unsqueeze(-1)
            if history_t.ndim == 3 and history_t.shape[-1] == 1:
                history_t = history_t.squeeze(-1)
            if history_t.ndim != 2:
                raise ValueError("signal_history must be rank-2 [time, batch]")
            if history_t.shape[1] != signal_t.reshape(-1).shape[0]:
                raise ValueError("signal_history batch dimension must match terminal_signal")
            history_t = history_t[:history_len]
            if history_t.shape[0] > 0:
                hist_enabled = history_t.index_select(1, enabled_idx)
                n_hist = int(hist_enabled.shape[0])
                sorted_history, _ = torch.sort(hist_enabled, dim=0)
                tail_prob = torch.clamp(prob_enabled * 0.5, min=0.0, max=0.5)
                tail_count_real = tail_prob * float(n_hist + 1)
                tail_count_floor = torch.floor(tail_count_real).to(dtype=torch.long)
                tail_count_frac = torch.clamp(
                    tail_count_real - tail_count_floor.to(dtype=signal_t.dtype),
                    min=0.0,
                    max=1.0,
                )
                _record(bucket, "terminal_tail.history_prepare_and_sort", time.perf_counter() - t0)

                t0 = time.perf_counter()
                resolvable = tail_count_floor > 0
                if bool(resolvable.any().item()):
                    history_idx = torch.nonzero(resolvable, as_tuple=False).reshape(-1)
                    signal_hist = signal_enabled.index_select(0, history_idx)
                    draw_hist = draw_enabled.index_select(0, history_idx)
                    tail_floor_hist = tail_count_floor.index_select(0, history_idx)
                    tail_frac_hist = tail_count_frac.index_select(0, history_idx)
                    sorted_hist = sorted_history.index_select(1, history_idx)
                    cols = torch.arange(history_idx.numel(), device=signal_t.device, dtype=torch.long)

                    lower_cut = sorted_hist[tail_floor_hist - 1, cols]
                    lower_strict = signal_hist < lower_cut
                    lower_boundary = torch.zeros_like(lower_strict)
                    has_lower_boundary = tail_floor_hist < n_hist
                    if bool(has_lower_boundary.any().item()):
                        hb_idx = torch.nonzero(has_lower_boundary, as_tuple=False).reshape(-1)
                        next_lower_cut = sorted_hist[
                            tail_floor_hist.index_select(0, hb_idx),
                            cols.index_select(0, hb_idx),
                        ]
                        lower_boundary.index_copy_(
                            0,
                            hb_idx,
                            (
                                (signal_hist.index_select(0, hb_idx) >= lower_cut.index_select(0, hb_idx))
                                & (signal_hist.index_select(0, hb_idx) < next_lower_cut)
                                & (draw_hist.index_select(0, hb_idx) < tail_frac_hist.index_select(0, hb_idx))
                            ),
                        )
                    _record(bucket, "terminal_tail.lower_tail_boundary", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    upper_cut = sorted_hist[n_hist - tail_floor_hist, cols]
                    upper_strict = signal_hist > upper_cut
                    upper_boundary = torch.zeros_like(upper_strict)
                    has_upper_boundary = tail_floor_hist < n_hist
                    if bool(has_upper_boundary.any().item()):
                        hb_idx = torch.nonzero(has_upper_boundary, as_tuple=False).reshape(-1)
                        prev_upper_cut = sorted_hist[
                            (n_hist - tail_floor_hist.index_select(0, hb_idx) - 1),
                            cols.index_select(0, hb_idx),
                        ]
                        upper_boundary.index_copy_(
                            0,
                            hb_idx,
                            (
                                (signal_hist.index_select(0, hb_idx) <= upper_cut.index_select(0, hb_idx))
                                & (signal_hist.index_select(0, hb_idx) > prev_upper_cut)
                                & (draw_hist.index_select(0, hb_idx) < tail_frac_hist.index_select(0, hb_idx))
                            ),
                        )
                    _record(bucket, "terminal_tail.upper_tail_boundary", time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    history_result = lower_strict | lower_boundary | upper_strict | upper_boundary
                    terminal_enabled.index_copy_(0, history_idx, history_result)
                    _record(bucket, "terminal_tail.scatter_history_result", time.perf_counter() - t0)

        t0 = time.perf_counter()
        terminal_out = torch.zeros_like(signal_t.reshape(-1), dtype=torch.bool)
        terminal_out.index_copy_(0, enabled_idx, terminal_enabled)
        out = terminal_out.reshape(signal_t.shape) & enabled_t
        _record(bucket, "terminal_tail.finalize_mask", time.perf_counter() - t0)
        _record(bucket, "terminal_tail.total", time.perf_counter() - t_total)
        return out

    return _instrumented


def _build_instrumented_terminal_reset_fn(prior, bucket):
    original_bonus_scale = prior._terminal_bonus_scale_from_draw
    original_bonus_from_signal = prior._terminal_bonus_from_signal
    original_terminal_tail = prior._terminal_tail_event_from_signal

    def _instrumented(
        *,
        state_next,
        reward_next,
        terminal_draw,
        reset_prob,
        bonus_scale_draw,
        bonus_scale_min,
        bonus_scale_max,
        bonus_tanh_c,
        reset_state,
        enabled,
        terminal_signal,
        terminal_bonus_base=None,
        terminal_signal_history=None,
        history_index=None,
        return_aux=False,
    ):
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        enabled_t = enabled
        if not torch.is_tensor(enabled_t):
            enabled_t = torch.as_tensor(enabled_t, device=state_next.device, dtype=torch.bool)
        else:
            enabled_t = enabled_t.to(device=state_next.device, dtype=torch.bool)
        if not bool(enabled_t.any().item()):
            terminal_zero = torch.zeros_like(reward_next, dtype=reward_next.dtype)
            _record(bucket, "terminal_reset.setup_enabled_prob", time.perf_counter() - t0)
            _record(bucket, "terminal_reset.total", time.perf_counter() - t_total)
            if not return_aux:
                return state_next, reward_next, terminal_zero
            aux = {
                "terminal_bonus_applied": torch.zeros_like(reward_next, dtype=reward_next.dtype),
                "terminal_bonus_event": torch.zeros_like(reward_next, dtype=reward_next.dtype),
                "terminal_event_mask": torch.zeros_like(reward_next, dtype=torch.bool),
            }
            return state_next, reward_next, terminal_zero, aux
        if terminal_signal is None:
            raise ValueError("terminal_signal must be provided when terminal reset is enabled")
        prob_t = torch.as_tensor(reset_prob, device=state_next.device, dtype=state_next.dtype)
        while prob_t.ndim < reward_next.ndim:
            prob_t = prob_t.unsqueeze(0)
        _record(bucket, "terminal_reset.setup_enabled_prob", time.perf_counter() - t0)

        t0 = time.perf_counter()
        bonus_scale_t = original_bonus_scale(
            bonus_scale_draw.to(device=state_next.device, dtype=state_next.dtype),
            scale_min=bonus_scale_min,
            scale_max=bonus_scale_max,
        )
        terminal_signal_t = terminal_signal.to(device=state_next.device, dtype=state_next.dtype)
        if terminal_bonus_base is None:
            terminal_bonus_t = original_bonus_from_signal(
                terminal_signal_t,
                bonus_scale=bonus_scale_t,
                tanh_c=bonus_tanh_c,
            )
        else:
            terminal_bonus_t = bonus_scale_t * terminal_bonus_base.to(
                device=state_next.device,
                dtype=state_next.dtype,
            )
        terminal_signal_event = terminal_signal_t.detach()
        _record(bucket, "terminal_reset.bonus_compute", time.perf_counter() - t0)

        t0 = time.perf_counter()
        terminal_next = original_terminal_tail(
            terminal_signal_event,
            enabled=enabled_t,
            reset_prob=prob_t,
            terminal_draw=terminal_draw,
            signal_history=terminal_signal_history,
            history_length=history_index,
        )
        _record(bucket, "terminal_reset.terminal_event_call", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if terminal_signal_history is not None and history_index is not None:
            hist_t = terminal_signal_history
            if not torch.is_tensor(hist_t):
                raise ValueError("terminal_signal_history must be a tensor")
            signal_store = terminal_signal_event.reshape(-1).to(device=hist_t.device, dtype=hist_t.dtype)
            hist_t[history_index].copy_(signal_store)
        _record(bucket, "terminal_reset.history_store", time.perf_counter() - t0)

        t0 = time.perf_counter()
        terminal_bonus_t = terminal_bonus_t.to(device=state_next.device, dtype=reward_next.dtype)
        terminal_bonus_applied = terminal_bonus_t * terminal_next.to(dtype=reward_next.dtype)
        reward_next = reward_next + terminal_bonus_applied
        _record(bucket, "terminal_reset.reward_bonus_apply", time.perf_counter() - t0)

        t0 = time.perf_counter()
        terminal_mask = terminal_next
        if terminal_mask.ndim < state_next.ndim:
            terminal_mask = terminal_mask.unsqueeze(-1)
        state_next = torch.where(
            terminal_mask,
            reset_state.to(device=state_next.device, dtype=state_next.dtype),
            state_next,
        )
        terminal_next_out = terminal_next.to(dtype=reward_next.dtype)
        _record(bucket, "terminal_reset.state_reset_where", time.perf_counter() - t0)
        _record(bucket, "terminal_reset.total", time.perf_counter() - t_total)

        if not return_aux:
            return state_next, reward_next, terminal_next_out
        aux = {
            "terminal_bonus_applied": terminal_bonus_applied.detach(),
            "terminal_bonus_event": terminal_bonus_t.detach(),
            "terminal_event_mask": terminal_next.to(dtype=torch.bool).detach(),
        }
        return state_next, reward_next, terminal_next_out, aux

    return _instrumented


def _summarize(bucket: dict[str, dict[str, float]], wall_s: float) -> dict[str, dict[str, float]]:
    out = {}
    for name, rec in sorted(bucket.items()):
        calls = int(rec["calls"])
        total_s = float(rec["total_s"])
        out[name] = {
            "calls": calls,
            "total_s": total_s,
            "mean_s": (total_s / calls) if calls > 0 else 0.0,
            "share_of_wall": (total_s / wall_s) if wall_s > 0 else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-env subpath profile for Phase 3 post-train/post-heldout serial eval."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--post-policy-bundle-path", type=str, required=True)
    parser.add_argument("--suite-target", type=str, choices=["train", "heldout"], required=True)
    parser.add_argument("--env-index", type=int, required=True)
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
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
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
    )
    algo.policy.load_state_dict(bundle["policy_state_dict"], strict=True)

    suite_target = str(args.suite_target).strip().lower()
    suite = train_suite if suite_target == "train" else heldout_suite
    suite_summary = train_suite_summary if suite_target == "train" else heldout_suite_summary
    single_suite = _single_env_suite(suite, int(args.env_index))
    policy_step_fn = _build_audit_ppo_policy_step_fn(
        algo.policy,
        sampled=True,
        single_eval_pos=int(args.single_eval_pos),
        boundary_contract_mode="normal",
    )

    bucket: dict[str, dict[str, float]] = {}
    with _replace_method(
        algo.policy,
        "_build_obs_from_rollout_step_inputs",
        _build_instrumented_obs_fn(algo.policy, bucket),
    ):
        with _replace_method(
            eval_prior,
            "_terminal_tail_event_from_signal",
            _build_instrumented_terminal_tail_fn(eval_prior, bucket),
        ):
            with _replace_method(
                eval_prior,
                "_apply_terminal_reset_step",
                _build_instrumented_terminal_reset_fn(eval_prior, bucket),
            ):
                t0 = time.perf_counter()
                profile = collect_suite_rewards_serial_profiled(
                    eval_prior,
                    suite=single_suite,
                    policy_step_fn=policy_step_fn,
                    n_samples=int(args.n_samples),
                    num_features=int(num_features),
                    single_eval_pos=int(args.single_eval_pos),
                    device=device_obj,
                    max_envs=1,
                )
                wall_s = float(time.perf_counter() - t0)
    vec_env.close()

    report = {
        "audit_entry": "phase3_post_eval_subpath_profile",
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
            "env_index": int(args.env_index),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "rollout_backend": str(args.rollout_backend),
            "train_profile": str(profile_cfg["train_profile"]),
            "profile_build_n_envs": 1,
            "profile_build_n_steps": 1,
            "profile_build_batch_size": 1,
        },
        "suite_summary": suite_summary,
        "bundle_contract": dict(bundle["contract"]),
        "profile": profile,
        "subpaths": _summarize(bucket, wall_s),
        "wall_s": wall_s,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
