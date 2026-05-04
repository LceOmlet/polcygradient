import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg, _make_deterministic_batch_plan
from ticl.analysis.fixed_env_h import build_fixed_env_h
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clone_h_repeated(h, count: int):
    return [copy.deepcopy(h) for _ in range(int(count))]


def _flatten_policy_grad(policy: torch.nn.Module) -> torch.Tensor:
    flat_parts = []
    for param in policy.parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat_parts.append(grad.detach().reshape(-1))
    if not flat_parts:
        return torch.zeros((0,), dtype=torch.float32)
    flat = torch.cat(flat_parts)
    return flat.to(dtype=torch.float32)


def _build_fixed_env_h(*, prior_cfg: dict, frozen_h_seed: int):
    return build_fixed_env_h(prior_cfg=prior_cfg, frozen_h_seed=int(frozen_h_seed))


def _build_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
):
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
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
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        verbose=0,
    )
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(n_steps))
    return algo, callback, vec_env, prior, num_features


def _collect_and_measure_policy_grad(algo, callback, vec_env) -> dict:
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
    clip_range = algo.clip_range(algo._current_progress_remaining)
    log_prob = eval_outputs["log_prob"]
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    clipped_objective = torch.min(policy_loss_1, policy_loss_2)
    policy_num = (
        clipped_objective
        * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    ).sum()
    policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

    algo.policy.optimizer.zero_grad(set_to_none=True)
    t_backward0 = time.perf_counter()
    policy_loss.backward()
    if algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    backward_wall_s = float(time.perf_counter() - t_backward0)
    grad_flat = _flatten_policy_grad(algo.policy)
    approx_kl = torch.mean((torch.exp(log_prob - rollout_data.old_log_prob) - 1) - (log_prob - rollout_data.old_log_prob))
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).to(dtype=ratio.dtype))
    returns_std = rollout_data.returns.detach().std().item() if rollout_data.returns.numel() > 1 else 0.0
    advantages_std = advantages.detach().std().item() if advantages.numel() > 1 else 0.0

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "approx_kl": float(approx_kl.detach().cpu().item()),
        "clip_fraction": float(clip_fraction.detach().cpu().item()),
        "objective_total": int(objective_mask.sum().detach().cpu().item()),
        "returns_std": float(returns_std),
        "advantages_std": float(advantages_std),
        "sep_state_reset_count": int(getattr(algo, "_rwkv_last_sep_state_reset_count", 0) or 0),
        "grad_norm": float(torch.linalg.vector_norm(grad_flat).detach().cpu().item()),
        "collect_wall_s": collect_wall_s,
        "backward_wall_s": backward_wall_s,
        "grad_vector": grad_flat.detach().cpu(),
    }


def run_sep_state_reset_gradient_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    reports = {}
    grad_vectors = {}
    for mode_name, reset_state in (("normal", False), ("state_reset_at_sep", True)):
        algo, callback, vec_env, prior, num_features = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            ppo_reset_env_state_at_sep=bool(reset_state),
        )
        try:
            report = _collect_and_measure_policy_grad(algo, callback, vec_env)
            grad_vectors[mode_name] = report.pop("grad_vector")
            report["num_features"] = int(num_features)
            report["ppo_reset_env_state_at_sep"] = bool(reset_state)
            reports[mode_name] = report
        finally:
            vec_env.close()
            del algo, callback, vec_env, prior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    normal_grad = grad_vectors["normal"]
    reset_grad = grad_vectors["state_reset_at_sep"]
    cosine = torch.nn.functional.cosine_similarity(
        normal_grad.unsqueeze(0),
        reset_grad.unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).item()
    grad_delta = reset_grad - normal_grad
    report = {
        "audit_entry": "sep_state_reset_gradient_compare",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "modes": reports,
        "grad_compare": {
            "cosine_similarity": float(cosine),
            "l2_delta_norm": float(torch.linalg.vector_norm(grad_delta).item()),
            "normal_grad_norm": float(torch.linalg.vector_norm(normal_grad).item()),
            "state_reset_grad_norm": float(torch.linalg.vector_norm(reset_grad).item()),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Fixed-environment gradient compare for SEP state-reset candidate")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    report = run_sep_state_reset_gradient_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )

    output_path = args.output_json
    if output_path is None:
        output_path = "/home/chen/RLPFN/artifacts/sep_state_reset_gradient_compare.json"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
