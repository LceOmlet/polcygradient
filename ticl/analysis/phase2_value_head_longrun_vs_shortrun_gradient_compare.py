import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _apply_boundary_contract_mode_flags,
    _apply_sep_state_reset_flag,
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.analysis.critic_value_fit_probe import _clone_h_repeated
from ticl.analysis.phase2_legacy_bar_longrun_probe import _build_algo as _build_longrun_algo
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo
from stable_baselines3.common.logger import configure as configure_logger


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _flatten_policy_grad(policy: torch.nn.Module) -> torch.Tensor:
    flat_parts = []
    for param in policy.parameters():
        if not bool(param.requires_grad):
            continue
        grad = param.grad
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat_parts.append(grad.detach().reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(flat_parts).to(dtype=torch.float32)


def _max_abs_diff_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    if int(a.numel()) == 0 and int(b.numel()) == 0:
        return 0.0
    return float((a - b).abs().max().detach().cpu().item())


def _build_shortrun_algo(
    *,
    checkpoint_path: str,
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
):
    _seed_all(int(build_seed))
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
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
        gamma=float(config["optimizer"].get("ppo_gamma", 1.0)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.95)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        actor_gae_space="normalized",
        actor_baseline_mode=str(actor_baseline_mode),
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=False,
        value_head_impl=str(value_head_impl),
        value_path_adapter_impl=str(value_path_adapter_impl),
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(train_env_seed)],
        rollout_rng_seeds=[int(train_rollout_seed)],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=bool(strict_native_rollout),
        restore_validation_policy_state=False,
        verbose=0,
    )
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


def _collect_and_measure_total_grad_generic(algo, callback, vec_env) -> dict[str, Any]:
    t_collect0 = time.perf_counter()
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    collect_wall_s = float(time.perf_counter() - t_collect0)

    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    advantages = _resolve_actor_advantages(rollout_data)

    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    value_logits = eval_outputs["value_logits"]
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]

    clip_range = algo.clip_range(algo._current_progress_remaining)
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    clipped_objective = torch.min(policy_loss_1, policy_loss_2)
    policy_num = (
        clipped_objective
        * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    ).sum()
    policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

    value_loss = algo.policy.compute_value_loss_from_logits(
        value_logits=value_logits,
        targets=rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
        objective_mask=objective_mask,
        value_target_bucket_idx=getattr(rollout_data, "value_target_bucket_idx", None),
    )

    if entropy is None:
        entropy_terms = -log_prob
    else:
        entropy_terms = entropy
    entropy_num = (
        entropy_terms
        * objective_mask.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    ).sum()
    entropy_loss = -(entropy_num / valid_total.to(device=entropy_num.device, dtype=entropy_num.dtype))

    total_loss = policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss
    algo.policy.optimizer.zero_grad(set_to_none=True)
    t_backward0 = time.perf_counter()
    total_loss.backward()
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    backward_wall_s = float(time.perf_counter() - t_backward0)
    grad_flat = _flatten_policy_grad(algo.policy)

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "returns_std": float(rollout_data.returns.detach().std().item() if rollout_data.returns.numel() > 1 else 0.0),
        "advantages_std": float(advantages.detach().std().item() if advantages.numel() > 1 else 0.0),
        "sep_state_reset_count": int(getattr(algo, "_rwkv_last_sep_state_reset_count", 0) or 0),
        "grad_norm": float(torch.linalg.vector_norm(grad_flat).detach().cpu().item()),
        "collect_wall_s": collect_wall_s,
        "backward_wall_s": backward_wall_s,
        "grad_vector": grad_flat.detach().cpu(),
    }


def run_phase2_value_head_longrun_vs_shortrun_gradient_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    strict_native_rollout: bool = False,
    value_head_impl: str = "legacy_bar",
    value_path_adapter_impl: str = "none",
) -> dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    num_features = int(config["prior"]["num_features"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    frozen_h = fixed_bundle["frozen_h"]

    reports = {}
    for mode_name in ("learned", "zero"):
        longrun_algo, longrun_callback, longrun_vec_env = _build_longrun_algo(
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
            actor_baseline_mode=str(mode_name),
            build_seed=int(build_seed),
            strict_native_rollout=bool(strict_native_rollout),
            value_head_impl=str(value_head_impl),
            value_path_adapter_impl=str(value_path_adapter_impl),
        )
        shortrun_algo, shortrun_callback, shortrun_vec_env = _build_shortrun_algo(
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
            actor_baseline_mode=str(mode_name),
            build_seed=int(build_seed),
            strict_native_rollout=bool(strict_native_rollout),
            value_head_impl=str(value_head_impl),
            value_path_adapter_impl=str(value_path_adapter_impl),
        )
        try:
            longrun_measure = _collect_and_measure_total_grad_generic(longrun_algo, longrun_callback, longrun_vec_env)
            shortrun_measure = _collect_and_measure_total_grad_generic(shortrun_algo, shortrun_callback, shortrun_vec_env)
            grad_a = longrun_measure.pop("grad_vector")
            grad_b = shortrun_measure.pop("grad_vector")
            grad_delta = grad_b - grad_a
            cosine = torch.nn.functional.cosine_similarity(
                grad_a.unsqueeze(0),
                grad_b.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()
            reports[str(mode_name)] = {
                "longrun_measure": longrun_measure,
                "shortrun_measure": shortrun_measure,
                "grad_compare": {
                    "cosine_similarity": float(cosine),
                    "l2_delta_norm": float(torch.linalg.vector_norm(grad_delta).item()),
                    "max_abs_grad_delta": _max_abs_diff_tensor(grad_a, grad_b),
                    "longrun_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
                    "shortrun_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
                    "exact_match": bool(_max_abs_diff_tensor(grad_a, grad_b) <= 0.0),
                },
            }
        finally:
            longrun_vec_env.close()
            shortrun_vec_env.close()
            del longrun_algo, longrun_callback, longrun_vec_env
            del shortrun_algo, shortrun_callback, shortrun_vec_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "audit_entry": "phase2_value_head_longrun_vs_shortrun_gradient_compare",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "contract": {
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "strict_native_rollout": bool(strict_native_rollout),
            "ppo_separate_value_backbone": False,
            "value_head_impl": str(value_head_impl),
            "value_path_adapter_impl": str(value_path_adapter_impl),
        },
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "modes": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gradient compare between short-run and long-run Phase 2 builds for a given value head candidate."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--strict-native-rollout", action="store_true")
    parser.add_argument("--value-head-impl", type=str, default="legacy_bar")
    parser.add_argument("--value-path-adapter-impl", type=str, default="none")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_value_head_longrun_vs_shortrun_gradient_compare.json",
    )
    args = parser.parse_args()

    report = run_phase2_value_head_longrun_vs_shortrun_gradient_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        strict_native_rollout=bool(args.strict_native_rollout),
        value_head_impl=str(args.value_head_impl),
        value_path_adapter_impl=str(args.value_path_adapter_impl),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
